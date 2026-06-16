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

"""Tests for the NexArm serial motor bus driver.

These tests exercise frame building/parsing, leader-to-follower mapping,
and the NexArmMotorsBus class using mock serial I/O — no physical hardware
is required.
"""

import struct
from unittest.mock import MagicMock, patch

import pytest

serial = pytest.importorskip("serial", reason="pyserial is required for NexArm tests")

from lerobot.motors.nexarm.nexarm import (  # noqa: E402
    CMD_LEROBOT_MODE,
    CMD_READ_POS,
    CMD_TORQUE,
    CMD_WRITE_POS,
    DEFAULT_BAUDRATE,
    JOINT_COUNT,
    JOINT_NAMES,
    POSITION_MAX,
    POSITION_MIN,
    SYSTEM_ID,
    NexArmMotorsBus,
    build_frame,
    map_leader_to_follower,
    parse_frame,
)

# ── Frame building tests ───────────────────────────────────────────────


class TestBuildFrame:
    """Tests for build_frame() — CommProtocol frame construction."""

    def test_header_bytes(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        assert frame[:2] == b"\xff\xff", "Frame must start with 0xFF 0xFF header"

    def test_device_id(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        assert frame[2] == 0xFF, "Device ID byte must match"

    def test_length_no_args(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        # length = len(args) + 2 = 0 + 2 = 2
        assert frame[3] == 2

    def test_length_with_args(self):
        args = bytes([1, 2, 3])
        frame = build_frame(0xFF, CMD_READ_POS, args)
        # length = len(args) + 2 = 3 + 2 = 5
        assert frame[3] == 5

    def test_cmd_byte(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        assert frame[4] == CMD_READ_POS

    def test_checksum_correct(self):
        frame = build_frame(0xFF, CMD_READ_POS)
        # checksum = ~(sum of id, length, cmd)) & 0xFF
        data_sum = frame[2] + frame[3] + frame[4]
        expected_checksum = (~data_sum) & 0xFF
        assert frame[-1] == expected_checksum

    def test_checksum_with_args(self):
        args = bytes([0x01])
        frame = build_frame(0xFF, CMD_TORQUE, args)
        data_sum = sum(frame[2:-1])
        expected_checksum = (~data_sum) & 0xFF
        assert frame[-1] == expected_checksum

    def test_round_trip_no_args(self):
        frame = build_frame(SYSTEM_ID, CMD_READ_POS)
        result = parse_frame(frame)
        assert result is not None
        device_id, cmd, args = result
        assert device_id == SYSTEM_ID
        assert cmd == CMD_READ_POS
        assert args == b""

    def test_round_trip_with_args(self):
        payload = bytes([0x01, 0x02, 0x03])
        frame = build_frame(SYSTEM_ID, CMD_TORQUE, payload)
        result = parse_frame(frame)
        assert result is not None
        device_id, cmd, args = result
        assert device_id == SYSTEM_ID
        assert cmd == CMD_TORQUE
        assert args == payload

    @pytest.mark.parametrize("cmd", [CMD_READ_POS, CMD_WRITE_POS, CMD_TORQUE, CMD_LEROBOT_MODE])
    def test_all_commands(self, cmd):
        frame = build_frame(SYSTEM_ID, cmd)
        result = parse_frame(frame)
        assert result is not None
        assert result[1] == cmd


# ── Frame parsing tests ────────────────────────────────────────────────


class TestParseFrame:
    """Tests for parse_frame() — CommProtocol frame parsing."""

    def test_too_short(self):
        assert parse_frame(b"\xff\xff\x01") is None

    def test_bad_header(self):
        frame = build_frame(SYSTEM_ID, CMD_READ_POS)
        bad_frame = b"\x00\x00" + frame[2:]
        assert parse_frame(bad_frame) is None

    def test_truncated_data(self):
        frame = build_frame(SYSTEM_ID, CMD_READ_POS, bytes(12))
        # Chop off last 3 bytes
        assert parse_frame(frame[:-3]) is None

    def test_bad_checksum(self):
        frame = bytearray(build_frame(SYSTEM_ID, CMD_READ_POS))
        frame[-1] ^= 0xFF  # corrupt checksum
        assert parse_frame(bytes(frame)) is None

    def test_firmware_buggy_checksum_accepted(self):
        """The leader firmware has a known checksum bug: it uses
        rx_packet.elements.length instead of tx_packet.elements.length
        when computing the checksum. The parser must accept both.
        """
        # Build a frame with 12 bytes of position data
        args = bytes(12)
        device_id = SYSTEM_ID
        cmd = CMD_READ_POS
        length = len(args) + 2

        # Build raw frame without checksum
        data_raw = bytes([device_id & 0xFF, length & 0xFF, cmd & 0xFF]) + args

        # Correct (full-range) checksum
        correct_checksum = (~sum(data_raw)) & 0xFF

        # Buggy (short) checksum — only id + length + cmd, ignoring args
        buggy_checksum = (~sum(data_raw[:3])) & 0xFF

        # Correct checksum must work
        frame_correct = b"\xff\xff" + data_raw + bytes([correct_checksum])
        assert parse_frame(frame_correct) is not None

        # Buggy checksum must also be accepted
        if buggy_checksum != correct_checksum:
            frame_buggy = b"\xff\xff" + data_raw + bytes([buggy_checksum])
            assert parse_frame(frame_buggy) is not None

    def test_parse_position_reply(self):
        """Parse a realistic CMD 96 position reply with 6 joint positions."""
        positions = [2048, 1024, 3072, 512, 4000, 2000]
        args = b""
        for p in positions:
            args += struct.pack("<h", p)

        frame = build_frame(SYSTEM_ID, CMD_READ_POS, args)
        result = parse_frame(frame)
        assert result is not None
        _, cmd, parsed_args = result
        assert cmd == CMD_READ_POS

        parsed_positions = [struct.unpack_from("<h", parsed_args, i * 2)[0] for i in range(JOINT_COUNT)]
        assert parsed_positions == positions


# ── Leader-to-follower mapping tests ───────────────────────────────────


class TestMapLeaderToFollower:
    """Tests for map_leader_to_follower() — leader-to-follower position mapping."""

    def test_output_length(self):
        leader_pos = [2048] * JOINT_COUNT
        result = map_leader_to_follower(leader_pos)
        assert len(result) == JOINT_COUNT

    def test_all_integers(self):
        leader_pos = [2048.7, 1024.3, 3000.5, 500.1, 4000.9, 2048.0]
        result = map_leader_to_follower(leader_pos)
        assert all(isinstance(v, int) for v in result)

    def test_joint2_mirrored(self):
        """Joint 2 (shoulder_lift, index 1) should be mirrored: 4096 - pos."""
        leader_pos = [2048, 1000, 2048, 2048, 2048, 2048]
        result = map_leader_to_follower(leader_pos)
        assert result[1] == 4096 - 1000

    def test_joint2_mirror_midpoint(self):
        """At midpoint (2048), mirror should give 2048."""
        leader_pos = [2048, 2048, 2048, 2048, 2048, 2048]
        result = map_leader_to_follower(leader_pos)
        assert result[1] == 2048

    def test_joint6_gripper_center(self):
        """At center (2048), gripper should map to 2833 + (2048-2048)*4 = 2833."""
        leader_pos = [2048, 2048, 2048, 2048, 2048, 2048]
        result = map_leader_to_follower(leader_pos)
        assert result[5] == 2833

    def test_joint6_gripper_clamp_high(self):
        """Gripper must be clamped to max 2833."""
        leader_pos = [2048, 2048, 2048, 2048, 2048, 1500]
        result = map_leader_to_follower(leader_pos)
        assert result[5] <= 2833

    def test_joint6_gripper_clamp_low(self):
        """Gripper must be clamped to min 1195."""
        leader_pos = [2048, 2048, 2048, 2048, 2048, 2500]
        result = map_leader_to_follower(leader_pos)
        assert result[5] >= 1195

    def test_passthrough_joints(self):
        """Joints 1, 3, 4, 5 (indices 0, 2, 3, 4) should pass through unchanged."""
        leader_pos = [100, 2048, 300, 400, 500, 2048]
        result = map_leader_to_follower(leader_pos)
        assert result[0] == 100
        assert result[2] == 300
        assert result[3] == 400
        assert result[4] == 500

    def test_clamp_to_position_range(self):
        """All output positions must be within [POSITION_MIN, POSITION_MAX]."""
        leader_pos = [-100, 5000, 2048, 2048, 2048, 2048]
        result = map_leader_to_follower(leader_pos)
        for p in result:
            assert POSITION_MIN <= p <= POSITION_MAX

    def test_float_inputs(self):
        """Float inputs should be rounded to integers."""
        leader_pos = [2048.6, 2048.4, 2048.5, 2048.0, 2048.9, 2048.1]
        result = map_leader_to_follower(leader_pos)
        assert all(isinstance(v, int) for v in result)


# ── NexArmMotorsBus tests (mock serial) ───────────────────────────────


def _make_mock_serial():
    """Create a MagicMock that mimics pyserial's Serial object."""
    mock = MagicMock()
    mock.is_open = True
    mock.in_waiting = 0
    mock.port = "/dev/null"
    return mock


class TestNexArmMotorsBus:
    """Tests for NexArmMotorsBus with mocked serial port."""

    def test_init(self):
        bus = NexArmMotorsBus(port="/dev/null")
        assert bus.port == "/dev/null"
        assert bus.baudrate == DEFAULT_BAUDRATE
        assert not bus.is_connected

    def test_connect_disconnect(self):
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")

            bus.connect()
            assert bus.is_connected
            mock_serial_cls.assert_called_once_with(
                port="/dev/null",
                baudrate=DEFAULT_BAUDRATE,
                timeout=bus.timeout,
                write_timeout=bus.timeout,
            )

            bus.disconnect()
            assert not bus.is_connected
            mock_ser.close.assert_called_once()

    def test_connect_idempotent(self):
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_serial_cls.return_value = _make_mock_serial()
            bus = NexArmMotorsBus(port="/dev/null")

            bus.connect()
            bus.connect()  # second call should be no-op
            assert mock_serial_cls.call_count == 1

    def test_disconnect_when_not_connected(self):
        bus = NexArmMotorsBus(port="/dev/null")
        bus.disconnect()  # should not raise

    def test_read_positions_success(self):
        """read_positions() should decode 6 int16 LE values from CMD 96 reply."""
        expected_positions = [2048, 1024, 3072, 512, 4000, 2000]

        # Build a realistic reply frame
        args = b""
        for p in expected_positions:
            args += struct.pack("<h", p)
        reply_frame = build_frame(SYSTEM_ID, CMD_READ_POS, args)

        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            # Simulate serial read returning the reply
            read_data = bytearray(reply_frame)
            mock_ser.in_waiting = len(read_data)
            mock_ser.read.return_value = bytes(read_data)

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            positions = bus.read_positions()

            assert positions == expected_positions

    def test_read_positions_timeout(self):
        """read_positions() should raise TimeoutError when no reply."""
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            # No data available
            mock_ser.in_waiting = 0
            mock_ser.read.return_value = b""

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()

            with pytest.raises(TimeoutError, match="No position reply"):
                bus.read_positions()

    def test_write_positions(self):
        """write_positions() should send a CMD 97 frame with 6 int16 LE values."""
        positions = [2048, 1024, 3072, 512, 4000, 2000]

        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.write_positions(positions)

            # Verify write was called
            mock_ser.write.assert_called()
            written = mock_ser.write.call_args[0][0]

            # Verify the frame is valid
            result = parse_frame(written)
            assert result is not None
            _, cmd, args = result
            assert cmd == CMD_WRITE_POS
            assert len(args) == JOINT_COUNT * 2

            # Verify position values
            parsed = [struct.unpack_from("<h", args, i * 2)[0] for i in range(JOINT_COUNT)]
            assert parsed == positions

    def test_write_positions_clamps(self):
        """write_positions() should clamp values to [0, 4095]."""
        positions = [-100, 5000, 2048, 2048, 2048, 2048]

        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.write_positions(positions)

            written = mock_ser.write.call_args[0][0]
            result = parse_frame(written)
            assert result is not None
            _, _, args = result

            parsed = [struct.unpack_from("<h", args, i * 2)[0] for i in range(JOINT_COUNT)]
            assert parsed[0] == POSITION_MIN  # -100 clamped to 0
            assert parsed[1] == POSITION_MAX  # 5000 clamped to 4095

    def test_set_torque_enable(self):
        """set_torque(True) should send CMD 98 with arg 0x01."""
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.set_torque(True)

            written = mock_ser.write.call_args[0][0]
            result = parse_frame(written)
            assert result is not None
            _, cmd, args = result
            assert cmd == CMD_TORQUE
            assert args == bytes([1])

    def test_set_torque_disable(self):
        """set_torque(False) should send CMD 98 with arg 0x00."""
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.set_torque(False)

            written = mock_ser.write.call_args[0][0]
            result = parse_frame(written)
            assert result is not None
            _, cmd, args = result
            assert cmd == CMD_TORQUE
            assert args == bytes([0])

    def test_enter_lerobot_mode(self):
        """enter_lerobot_mode() should send CMD 68 with arg 0x01."""
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.enter_lerobot_mode()

            written = mock_ser.write.call_args[0][0]
            result = parse_frame(written)
            assert result is not None
            _, cmd, args = result
            assert cmd == CMD_LEROBOT_MODE
            assert args == bytes([1])

    def test_exit_lerobot_mode(self):
        """exit_lerobot_mode() should send CMD 68 with arg 0x00."""
        with patch("lerobot.motors.nexarm.nexarm.serial.Serial") as mock_serial_cls:
            mock_ser = _make_mock_serial()
            mock_serial_cls.return_value = mock_ser

            bus = NexArmMotorsBus(port="/dev/null")
            bus.connect()
            bus.exit_lerobot_mode()

            written = mock_ser.write.call_args[0][0]
            result = parse_frame(written)
            assert result is not None
            _, cmd, args = result
            assert cmd == CMD_LEROBOT_MODE
            assert args == bytes([0])

    def test_send_raises_when_not_connected(self):
        """Sending a command when not connected should raise ConnectionError."""
        bus = NexArmMotorsBus(port="/dev/null")
        with pytest.raises(ConnectionError, match="Serial port not open"):
            bus.read_positions()


# ── Constant / naming tests ───────────────────────────────────────────


class TestConstants:
    """Verify NexArm constants are consistent."""

    def test_joint_count_matches_names(self):
        assert len(JOINT_NAMES) == JOINT_COUNT

    def test_joint_names_are_strings(self):
        for name in JOINT_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_position_range(self):
        assert POSITION_MIN == 0
        assert POSITION_MAX == 4095

    def test_joint_names_expected(self):
        expected = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        assert expected == JOINT_NAMES
