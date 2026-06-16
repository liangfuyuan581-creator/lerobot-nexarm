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

"""Tests for the NexArmLeader teleoperator class.

Exercises connect/disconnect lifecycle, get_action, leader-to-follower mapping,
and calibration using a mocked serial bus — no physical hardware required.
"""

from unittest.mock import MagicMock, patch

import pytest

serial = pytest.importorskip("serial", reason="pyserial is required for NexArm tests")

from lerobot.motors.nexarm.nexarm import JOINT_NAMES, POSITION_MAX, POSITION_MIN  # noqa: E402
from lerobot.teleoperators.nexarm_leader import NexArmLeader, NexArmLeaderConfig  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────


def _make_bus_mock(positions: list[int] | None = None) -> MagicMock:
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
    bus.set_torque.return_value = None

    return bus


@pytest.fixture
def leader():
    bus_mock = _make_bus_mock()

    with (
        patch(
            "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
            return_value=bus_mock,
        ),
        patch.object(NexArmLeader, "configure", lambda self: None),
    ):
        cfg = NexArmLeaderConfig(port="/dev/null")
        teleop = NexArmLeader(cfg)
        teleop._bus_mock = bus_mock
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


@pytest.fixture
def leader_with_positions():
    positions = [100, 200, 300, 400, 500, 600]
    bus_mock = _make_bus_mock(positions)

    with (
        patch(
            "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
            return_value=bus_mock,
        ),
        patch.object(NexArmLeader, "configure", lambda self: None),
    ):
        cfg = NexArmLeaderConfig(port="/dev/null")
        teleop = NexArmLeader(cfg)
        teleop._bus_mock = bus_mock
        teleop._raw_positions = positions
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


# ── Connect / Disconnect ─────────────────────────────────────────────


class TestConnectDisconnect:
    def test_not_connected_initially(self, leader):
        assert not leader.is_connected

    def test_connect(self, leader):
        leader.connect()
        assert leader.is_connected

    def test_disconnect(self, leader):
        leader.connect()
        leader.disconnect()
        assert not leader.is_connected

    def test_connect_idempotent(self, leader):
        from lerobot.utils.errors import DeviceAlreadyConnectedError

        leader.connect()
        with pytest.raises(DeviceAlreadyConnectedError):
            leader.connect()
        leader._bus_mock.connect.assert_called_once()

    def test_disconnect_when_not_connected(self, leader):
        from lerobot.utils.errors import DeviceNotConnectedError

        with pytest.raises(DeviceNotConnectedError):
            leader.disconnect()


# ── get_action ────────────────────────────────────────────────────────


class TestGetAction:
    def test_returns_all_joint_keys(self, leader_with_positions):
        teleop = leader_with_positions
        teleop.connect()
        action = teleop.get_action()

        expected_keys = {f"{name}.pos" for name in JOINT_NAMES}
        assert set(action.keys()) == expected_keys

    def test_values_are_floats(self, leader_with_positions):
        teleop = leader_with_positions
        teleop.connect()
        action = teleop.get_action()

        for key in action:
            assert isinstance(action[key], float)

    def test_shoulder_lift_is_mirrored(self, leader_with_positions):
        """shoulder_lift (idx 1) should be mirrored: 4096 - raw_pos."""
        teleop = leader_with_positions
        teleop.connect()
        action = teleop.get_action()

        raw = teleop._raw_positions[1]
        assert action["shoulder_lift.pos"] == pytest.approx(float(4096 - raw))

    def test_other_joints_pass_through(self, leader_with_positions):
        """Joints other than shoulder_lift and gripper pass through unchanged."""
        teleop = leader_with_positions
        teleop.connect()
        action = teleop.get_action()

        for idx, name in enumerate(JOINT_NAMES):
            if name in ("shoulder_lift", "gripper"):
                continue
            raw = teleop._raw_positions[idx]
            expected = float(max(POSITION_MIN, min(POSITION_MAX, raw)))
            assert action[f"{name}.pos"] == pytest.approx(expected)

    def test_gripper_remapped(self, leader_with_positions):
        """Gripper (idx 5) is remapped: 2833 + (pos - 2048) * 4, clamped [1195, 2833]."""
        teleop = leader_with_positions
        teleop.connect()
        action = teleop.get_action()

        raw = teleop._raw_positions[5]
        remapped = 2833 + (raw - 2048) * 4
        remapped = max(1195, min(2833, remapped))
        remapped = max(POSITION_MIN, min(POSITION_MAX, remapped))
        assert action["gripper.pos"] == pytest.approx(float(remapped))

    def test_calls_read_positions(self, leader):
        leader.connect()
        leader.get_action()
        leader._bus_mock.read_positions.assert_called_once()


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_default_baudrate(self):
        cfg = NexArmLeaderConfig(port="/dev/null")
        assert cfg.baudrate == 1_000_000

    def test_custom_port(self):
        cfg = NexArmLeaderConfig(port="COM18")
        assert cfg.port == "COM18"

    def test_action_features_match_joints(self):
        with patch(
            "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
        ):
            cfg = NexArmLeaderConfig(port="/dev/null")
            teleop = NexArmLeader(cfg)
            expected = {f"{name}.pos" for name in JOINT_NAMES}
            assert set(teleop.action_features.keys()) == expected

    def test_feedback_features_empty(self):
        with patch(
            "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
        ):
            cfg = NexArmLeaderConfig(port="/dev/null")
            teleop = NexArmLeader(cfg)
            assert teleop.feedback_features == {}
