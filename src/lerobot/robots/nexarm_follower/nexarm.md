# NexArm 6-DOF Robot Arm — LeRobot Integration Guide

NexArm is a high-performance 6-DOF desktop robot arm powered by an ESP32 + AT32F421 co-processor architecture, featuring HX-30HM serial bus servos with 12-bit precision and 1 Mbps communication. Designed for imitation learning research, it offers precise, responsive control at an accessible price point.

---

## Table of Contents

- [Hardware Overview](#hardware-overview)
- [Installation](#installation)
- [Step 1: Find Serial Ports](#step-1-find-serial-ports)
- [Step 2: Find Cameras](#step-2-find-cameras)
- [Step 3: Teleoperation Test](#step-3-teleoperation-test)
- [Step 4: Record a Dataset](#step-4-record-a-dataset)
- [Step 5: Train a Policy](#step-5-train-a-policy)
- [Step 6: Run Inference](#step-6-run-inference)
- [Code Architecture](#code-architecture)
- [Troubleshooting](#troubleshooting)

---

## Hardware Overview

### Components

| Component | Description |
|-----------|-------------|
| **Leader Arm** | ESP32 dev board directly driving 6 HX-30HM servos. Used for teleoperation — the operator freely drags this arm |
| **Follower Arm** | ESP32 dev board + AT32F421 co-processor driving 6 HX-30HM servos. Mirrors the leader or executes policy outputs |
| **Servos** | HX-30HM high-performance serial bus servos — 12-bit resolution (0–4095), 1 Mbps baud rate, high torque and fast response |
| **Cameras** | 2 USB cameras — "front" (workspace overview) and "wrist" (end-effector close-up), 640×480 @ 30 FPS |

### Joint Layout (6-DOF)

| Joint ID | Name | Description |
|----------|------|-------------|
| 1 | `shoulder_pan` | Base rotation |
| 2 | `shoulder_lift` | Mirrored between leader and follower (4096 − pos) |
| 3 | `elbow_flex` | Elbow joint |
| 4 | `wrist_flex` | Wrist pitch |
| 5 | `wrist_roll` | Wrist rotation |
| 6 | `gripper` | Gripper open/close, remapped range [1195, 2833] |

### Communication Protocol

NexArm uses a custom CommProtocol over USB serial:

```
Frame format: [0xFF][0xFF][ID][LEN][CMD][ARGS...][CHECKSUM]
```

| Command | Function | Direction |
|---------|----------|-----------|
| CMD 68 | Enter/exit LeRobot bridge mode (follower only) | Host → Follower |
| CMD 96 | Read 6 servo positions (reply: 12 bytes) | Host → Device → Host |
| CMD 97 | Write 6 servo positions (12 bytes, no reply) | Host → Device |
| CMD 98 | Enable/disable torque | Host → Device |

---

## Installation

### 1. Clone and Install

```bash
cd lerobot
pip install -e ".[nexarm]"
```

The `[nexarm]` extra automatically installs `pyserial`.

### 2. Connect Hardware

- Connect the **Follower** ESP32 to the computer via USB
- Connect the **Leader** ESP32 to the computer via USB
- Plug in both USB cameras (front + wrist)

---

## Step 1: Find Serial Ports

Identify which serial port corresponds to each arm:

```bash
python -m lerobot.scripts.lerobot_find_port
```

Example output:

| Platform | Leader Port | Follower Port |
|----------|-------------|---------------|
| Windows | `COM18` | `COM19` |
| Linux | `/dev/ttyUSB0` | `/dev/ttyUSB1` |
| macOS | `/dev/tty.usbserial-110` | `/dev/tty.usbserial-120` |

> **Tip:** If you cannot tell which port belongs to which arm, try plugging in only one arm at a time, or check your system's device manager.

---

## Step 2: Find Cameras

Determine which camera index corresponds to "front" and "wrist":

```bash
python -m lerobot.scripts.lerobot_find_cameras --camera-type opencv
```

Or manually scan and capture test frames:

```python
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"cam_{i}.png", frame)
            print(f"Camera {i}: available")
        cap.release()
```

Review the saved images to identify:
- **front**: overhead view of the workspace
- **wrist**: close-up of the end-effector / gripper

> **Note:** Camera indices may change when USB devices are plugged/unplugged. Re-scan after any hardware changes.

---

## Step 3: Teleoperation Test

Verify the leader-follower teleoperation link. The leader runs in torque-off mode so the operator can freely drag it; the follower mirrors in real time.

### Option A: Using a Config File

Edit `examples/nexarm/teleoperate.yaml` to match your serial ports and camera indices, then run:

```bash
python -m lerobot.scripts.lerobot_teleoperate \
    --config_path=examples/nexarm/teleoperate.yaml
```

### Option B: Using Command-Line Arguments

```bash
python -m lerobot.scripts.lerobot_teleoperate \
    --robot.type=nexarm_follower \
    --robot.port=COM19 \
    --teleop.type=nexarm_leader \
    --teleop.port=COM18 \
    --fps=30
```

### What to Check

- Dragging the leader arm should make the follower track in real time
- All joint directions should be consistent (shoulder_lift is automatically mirrored)
- Gripper open/close should map correctly

> **Tip:** If teleoperation feels laggy, check that no other program (e.g., Arduino IDE serial monitor) is occupying the COM port.

---

## Step 4: Record a Dataset

Record demonstration data via teleoperation for imitation learning training.

### Run Recording

Edit `examples/nexarm/record.yaml` with your settings, then:

```bash
python -m lerobot.scripts.lerobot_record \
    --config_path=examples/nexarm/record.yaml
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset.repo_id` | `${HF_USER}/nexarm_pick` | Dataset name (HuggingFace Hub format) |
| `dataset.num_episodes` | 50 | Number of episodes to record |
| `dataset.episode_time_s` | 10 | Duration per episode (seconds) |
| `dataset.reset_time_s` | 10 | Time for resetting the scene between episodes |
| `dataset.fps` | 30 | Recording frame rate |
| `dataset.push_to_hub` | true | Upload to HuggingFace Hub when done |

### Recording Workflow

1. The script connects to both arms and cameras
2. For each episode:
   - Press Enter to start recording
   - Drag the leader arm to perform the task (e.g., pick up an object)
   - Recording stops automatically after `episode_time_s`
   - Reset the scene during `reset_time_s`
3. After all episodes, the dataset is saved locally (and optionally uploaded to Hub)

### Local-Only Recording (No Upload)

```bash
python -m lerobot.scripts.lerobot_record \
    --config_path=examples/nexarm/record.yaml \
    --dataset.repo_id=local/nexarm_pick \
    --dataset.push_to_hub=false
```

### Data Quality Tips

- Record **at least 50 episodes** for reasonable training results
- Maintain consistency across episodes (similar starting positions and trajectories)
- Ensure good lighting and clear camera views
- Use `lerobot_dataset_viz` to review dataset quality after recording

---

## Step 5: Train a Policy

Train an ACT (Action Chunking with Transformers) policy on the recorded dataset:

```bash
python -m lerobot.scripts.lerobot_train \
    --config_path=examples/nexarm/train.yaml \
    --dataset.repo_id=${HF_USER}/nexarm_pick
```

### Training Recommendations

| Item | Recommended | Notes |
|------|-------------|-------|
| Hardware | CUDA GPU | A100 or RTX 3090+ recommended |
| Episodes | 50+ | More is better |
| Steps | 100,000 | Adjust based on loss convergence |
| Batch size | 8 | Adjust for your GPU memory |
| Save frequency | 25,000 | Checkpoint every 25k steps |

### Training Output

Checkpoints are saved to:

```
outputs/train/nexarm_act/last/pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

---

## Step 6: Run Inference

Deploy the trained policy on the real robot. The follower arm autonomously executes model-predicted actions — no leader arm needed.

```bash
python -m lerobot.scripts.lerobot_record \
    --config_path=examples/nexarm/inference.yaml \
    --policy.path=outputs/train/nexarm_act/last/pretrained_model
```

### Important Notes

- **No `--teleop` argument needed** — the policy replaces the human operator
- The dataset name must start with `eval_` (already configured in the YAML)
- Inference runs are recorded as a dataset for evaluation
- If no GPU is available, change `"device": "cuda"` to `"device": "cpu"` in the model's `config.json`

### Inference Performance

| Device | Approx. FPS | Notes |
|--------|-------------|-------|
| NVIDIA A100 | 30 Hz | Full real-time control |
| NVIDIA RTX 3090 | 25–30 Hz | Near real-time |
| CPU (i7) | 1–2 Hz | Too slow for real use; functional testing only |

> **GPU inference is strongly recommended.** CPU inference is too slow for responsive robot control.

---

## Code Architecture

### File Structure

```
src/lerobot/
├── motors/nexarm/                          # Serial bus driver layer
│   ├── __init__.py                         # Exports NexArmMotorsBus
│   └── nexarm.py                           # CommProtocol frame building/parsing,
│                                           #   position read/write, torque, bridge mode
├── robots/nexarm_follower/                 # Follower robot
│   ├── __init__.py                         # Exports NexArmFollower, NexArmFollowerConfig
│   ├── config_nexarm_follower.py           # RobotConfig subclass (port, cameras, baudrate)
│   └── nexarm_follower.py                  # Robot subclass (connect, observe, act)
└── teleoperators/nexarm_leader/            # Leader teleoperator
    ├── __init__.py                         # Exports NexArmLeader, NexArmLeaderConfig
    ├── config_nexarm_leader.py             # TeleoperatorConfig subclass (port, baudrate)
    └── nexarm_leader.py                    # Teleoperator subclass (read action, leader-to-follower mapping)
```

### Modified Upstream Files

| File | Change |
|------|--------|
| `src/lerobot/robots/utils.py` | Added `nexarm_follower` branch in `make_robot_from_config()` |
| `src/lerobot/teleoperators/utils.py` | Added `nexarm_leader` branch in `make_teleoperator_from_config()` |
| `pyproject.toml` | Added `nexarm = ["lerobot[pyserial-dep]"]` to optional dependencies |

### Data Flow

```
Teleoperation mode:
  Leader → read_positions() → map_leader_to_follower() → write_positions() → Follower

Policy inference mode:
  Cameras + Follower joint positions → ACT model → predicted action → write_positions() → Follower
```

---

## Troubleshooting

### Serial Port Permission Denied (Linux)

```bash
sudo usermod -a -G dialout $USER
# Log out and back in for the change to take effect
```

### Cameras Not Found

- Run `lerobot_find_cameras` to scan available indices
- Close other programs that may be using the cameras (OBS, browsers, etc.)
- Camera indices may change after USB replug — re-scan to confirm

### Follower Arm Not Moving During Teleoperation

1. Verify the follower's serial port is correct
2. Confirm the follower ESP32 firmware supports CMD 68 (LeRobot bridge mode)
3. Try power-cycling the follower arm

### Training Loss Not Decreasing

- Ensure you have enough demonstration episodes (50+ recommended)
- Check that camera frames in the dataset are normal (not black/blurry)
- Try increasing the learning rate (e.g., `--policy.optimizer_lr=1e-4`)

### CPU Inference Too Slow

- Use a CUDA GPU for real-time inference
- Or reduce camera resolution: `width=320, height=240`
- Or lower the frame rate: `--dataset.fps=5`

### Serial Checksum Errors

The leader firmware has a known checksum bug (`tx_packet_complete` uses `rx_packet.elements.length` instead of `tx_packet.elements.length`). The driver accepts both correct and incorrect checksums for backward compatibility. This does not affect functionality.
