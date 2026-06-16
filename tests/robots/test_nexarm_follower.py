#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the NexArmFollower robot class.

Exercises connect/disconnect lifecycle, get_observation, send_action,
and calibration using mocked serial and camera I/O — no physical hardware
is required.
"""

from unittest.mock import MagicMock, patch

import pytest

serial = pytest.importorskip("serial", reason="pyserial is required for NexArm tests")

from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.robots.nexarm_follower import (  # noqa: E402
    NexArmFollower,
    NexArmFollowerConfig,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _make_bus_mock(positions: list[int] | None = None) -> MagicMock:
    """Return a MagicMock that mimics NexArmMotorsBus.

    Parameters
    ----------
    positions : list[int] | None
        The 6 joint positions that ``read_positions()`` should return.
        Defaults to ``[2048] * 6`` (midpoint for all joints).
    """
    if positions is None:
        positions = [2048] * 6

    bus = MagicMock(name="NexArmBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect():
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.read_positions.return_value = list(positions)
    bus.write_positions.return_value = None
    bus.set_torque.return_value = None
    bus.enter_lerobot_mode.return_value = None
    bus.exit_lerobot_mode.return_value = None

    return bus


@pytest.fixture
def follower():
    """Create a NexArmFollower with fully mocked bus and no cameras."""
    bus_mock = _make_bus_mock()

    with (
        patch(
            "lerobot.robots.nexarm_follower.nexarm_follower.NexArmMotorsBus",
            return_value=bus_mock,
        ),
        patch.object(NexArmFollower, "configure", lambda self: None),
    ):
        cfg = NexArmFollowerConfig(port="/dev/null", cameras={})
        robot = NexArmFollower(cfg)
        # Expose mock for assertions
        robot._bus_mock = bus_mock
        yield robot
        if robot.is_connected:
            robot.disconnect()


@pytest.fixture
def follower_with_positions():
    """Create a NexArmFollower returning known position values (1-indexed)."""
    positions = [100, 200, 300, 400, 500, 600]
    bus_mock = _make_bus_mock(positions)

    with (
        patch(
            "lerobot.robots.nexarm_follower.nexarm_follower.NexArmMotorsBus",
            return_value=bus_mock,
        ),
        patch.object(NexArmFollower, "configure", lambda self: None),
    ):
        cfg = NexArmFollowerConfig(port="/dev/null", cameras={})
        robot = NexArmFollower(cfg)
        robot._bus_mock = bus_mock
        robot._expected_positions = positions
        yield robot
        if robot.is_connected:
            robot.disconnect()


# ── Connect / Disconnect ─────────────────────────────────────────────


class TestConnectDisconnect:
    """Lifecycle tests for NexArmFollower."""

    def test_not_connected_initially(self, follower):
        assert not follower.is_connected

    def test_connect(self, follower):
        follower.connect()
        assert follower.is_connected

    def test_connect_calls_enter_lerobot_mode(self, follower):
        follower.connect()
        follower._bus_mock.enter_lerobot_mode.assert_called_once()

    def test_disconnect(self, follower):
        follower.connect()
        follower.disconnect()
        assert not follower.is_connected

    def test_connect_idempotent(self, follower):
        from lerobot.utils.errors import DeviceAlreadyConnectedError

        follower.connect()
        with pytest.raises(DeviceAlreadyConnectedError):
            follower.connect()
        follower._bus_mock.connect.assert_called_once()

    def test_disconnect_when_not_connected(self, follower):
        from lerobot.utils.errors import DeviceNotConnectedError

        with pytest.raises(DeviceNotConnectedError):
            follower.disconnect()


# ── Observation ───────────────────────────────────────────────────────


class TestGetObservation:
    """Tests for NexArmFollower.get_observation()."""

    def test_returns_all_joint_keys(self, follower_with_positions):
        robot = follower_with_positions
        robot.connect()
        obs = robot.get_observation()

        expected_keys = {f"{name}.pos" for name in JOINT_NAMES}
        assert set(obs.keys()) == expected_keys

    def test_returns_correct_values(self, follower_with_positions):
        robot = follower_with_positions
        robot.connect()
        obs = robot.get_observation()

        for i, name in enumerate(JOINT_NAMES):
            assert obs[f"{name}.pos"] == float(robot._expected_positions[i])

    def test_values_are_floats(self, follower_with_positions):
        robot = follower_with_positions
        robot.connect()
        obs = robot.get_observation()

        for key in obs:
            assert isinstance(obs[key], float)


# ── Send Action ───────────────────────────────────────────────────────


class TestSendAction:
    """Tests for NexArmFollower.send_action()."""

    def test_calls_write_positions(self, follower):
        follower.connect()
        action = {f"{name}.pos": float(2048) for name in JOINT_NAMES}
        follower.send_action(action)

        follower._bus_mock.write_positions.assert_called_once()

    def test_passes_correct_values(self, follower):
        follower.connect()
        values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        action = {f"{name}.pos": val for name, val in zip(JOINT_NAMES, values, strict=True)}
        follower.send_action(action)

        expected_ints = [100, 200, 300, 400, 500, 600]
        follower._bus_mock.write_positions.assert_called_once_with(expected_ints)

    def test_clamps_out_of_range(self, follower):
        follower.connect()
        # Shoulder_pan below min, shoulder_lift above max
        action = {
            "shoulder_pan.pos": -100.0,
            "shoulder_lift.pos": 5000.0,
            "elbow_flex.pos": 2048.0,
            "wrist_flex.pos": 2048.0,
            "wrist_roll.pos": 2048.0,
            "gripper.pos": 2048.0,
        }
        follower.send_action(action)

        written = follower._bus_mock.write_positions.call_args[0][0]
        assert written[0] == 0  # clamped to POSITION_MIN
        assert written[1] == 4095  # clamped to POSITION_MAX

    def test_returns_feedback(self, follower_with_positions):
        robot = follower_with_positions
        robot.connect()
        action = {f"{name}.pos": float(2048) for name in JOINT_NAMES}
        result = robot.send_action(action)

        expected_keys = {f"{name}.pos" for name in JOINT_NAMES}
        assert set(result.keys()) == expected_keys

    def test_rounds_float_values(self, follower):
        follower.connect()
        action = {f"{name}.pos": 2048.7 for name in JOINT_NAMES}
        follower.send_action(action)

        written = follower._bus_mock.write_positions.call_args[0][0]
        # 2048.7 should round to 2049
        assert all(v == 2049 for v in written)


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    """Tests for NexArmFollowerConfig defaults."""

    def test_default_baudrate(self):
        cfg = NexArmFollowerConfig(port="/dev/null")
        assert cfg.baudrate == 1_000_000

    def test_disable_torque_on_disconnect_default(self):
        cfg = NexArmFollowerConfig(port="/dev/null")
        assert cfg.disable_torque_on_disconnect is True

    def test_custom_port(self):
        cfg = NexArmFollowerConfig(port="COM19")
        assert cfg.port == "COM19"

    def test_cameras_default_empty(self):
        cfg = NexArmFollowerConfig(port="/dev/null")
        # Default cameras should be a dict (may be empty or have defaults)
        assert isinstance(cfg.cameras, dict)
