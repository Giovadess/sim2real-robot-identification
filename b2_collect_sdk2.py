"""Collect B2 SysID data with native SDK2. Run only with the base securely fixed."""

from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

import collection_trajectories as trajectories
import config


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "robot_model/b2/scene_flat.xml"
SDK_TO_DATASET = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
DATASET_TO_SDK = np.argsort(SDK_TO_DATASET)
LOWER = np.tile([-0.87, -0.94, -2.82], 4)
UPPER = np.tile([0.87, 4.69, -0.43], 4)
STATE_TIMEOUT = 0.1
COMMAND_HZ = 500
RECORD_HZ = config.frequency_collection
RIG_CHECK_MAX_SPEED = 0.10  # rad/s, commanded reference
RIG_CHECK_HOLD = 8.0  # s
RIG_CHECK_MAX_MEASURED_SPEED = 0.5  # rad/s
RIG_CHECK_MAX_TRACKING_ERROR = 0.3  # rad
SIDEWAYS_LEG_START_HIP = 0.45
SIDEWAYS_LEG_END = np.array([0.65, 0.9, -2.0])  # outward hip sweep with moderate thigh/calf extension


def make_trajectory(model, mode: str, seed: int, speed: float):
    key1 = model.key("sys_id_1").qpos[7:10]
    key2 = model.key("sys_id_2").qpos[7:10]
    if mode == "chirp":
        return trajectories.quadruped_chirp_trajectory(key1, key2, RECORD_HZ, speed_scale=speed)
    start = trajectories.mirror_leg_to_quadruped(key1[None], True)[0]
    if mode == "sideways":
        side_start = key1.copy()
        side_start[0] = SIDEWAYS_LEG_START_HIP
        start = trajectories.mirror_leg_to_quadruped(side_start[None], True)[0]
        side = trajectories.quadruped_excitation_trajectory(
            side_start, SIDEWAYS_LEG_END, start, RECORD_HZ, seed=seed, speed_scale=speed,
        )
        return trajectories.JointTrajectory(
            side.positions, RECORD_HZ, "sideways",
            dict(side.metadata, sideways_profile=1), label="sideways",
        )
    return trajectories.quadruped_excitation_trajectory(
        key1, key2, start, RECORD_HZ, seed=seed, speed_scale=speed,
    )


def next_path(prefix: str, sim: bool) -> Path:
    directory = ROOT / ("datasets/b2_sim" if sim else "datasets/b2")
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while (directory / f"{prefix}_{index}.npz").exists() or (directory / f"{prefix}_{index}.pt").exists():
        index += 1
    return directory / f"{prefix}_{index}.npz"


def foot_bottoms(model, positions):
    """Model foot sphere bottoms relative to the fixed base origin, FL/FR/RL/RR."""
    data = mujoco.MjData(model)
    feet = [model.geom(leg).id for leg in ("FL", "FR", "RL", "RR")]
    radii = np.array([model.geom_size[foot, 0] for foot in feet])
    bottoms = np.empty((len(positions), 4))
    data.qpos[:7] = (0, 0, 0, 1, 0, 0, 0)
    for i, pose in enumerate(positions):
        data.qpos[7:19] = pose
        mujoco.mj_forward(model, data)
        bottoms[i] = data.geom_xpos[feet, 2] - radii
    return bottoms


def lowest_foot_poses(model, planned):
    """Find each leg's lowest foot pose within the exact planned joint samples."""
    bottoms = foot_bottoms(model, planned.positions)
    indices = np.argmin(bottoms, axis=0)
    target = np.empty(12)
    for leg, sample in enumerate(indices):
        sl = slice(3 * leg, 3 * leg + 3)
        target[sl] = planned.positions[sample, sl]
    target_bottom = float(foot_bottoms(model, target[None, :]).min())
    return target, float(bottoms.min()), target_bottom


def rig_check_trajectory(start, target, planned):
    distance = float(np.max(np.abs(target - start)))
    travel_time = max(8.0, 1.5 * distance / RIG_CHECK_MAX_SPEED)
    n = int(np.ceil(travel_time * RECORD_HZ))
    u = np.linspace(0.0, 1.0, n + 1)
    blend = 3 * u**2 - 2 * u**3
    outward = start + blend[:, None] * (target - start)
    hold = np.repeat(target[None, :], int(RIG_CHECK_HOLD * RECORD_HZ), axis=0)
    positions = np.concatenate((outward, hold, outward[-2::-1]), axis=0)
    metadata = dict(planned.metadata, rig_check=1, rig_check_max_speed=RIG_CHECK_MAX_SPEED)
    return trajectories.JointTrajectory(positions, RECORD_HZ, "rig_check", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp98s0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--mode", choices=("chirp", "excitation", "sideways"), default="excitation")
    parser.add_argument("--seed", type=int, default=trajectories.DEFAULT_SEED)
    parser.add_argument("--speed", type=float, default=0.3)
    parser.add_argument("--preview", action="store_true", help="Print trajectory; do not initialize DDS")
    parser.add_argument("--rig-check", action="store_true", help="Slowly reach each foot's lowest planned pose, hold, and return")
    parser.add_argument("--rest-foot-clearance", type=float, metavar="METERS",
                        help="Measured lowest foot-to-floor gap at rest; required for hardware motion")
    parser.add_argument("--min-clearance", type=float, default=0.15, metavar="METERS",
                        help="Minimum predicted foot-to-floor gap, at least 0.15 m (default: 0.15)")
    parser.add_argument("--sim", action="store_true", help="Use only DDS domain 1 on loopback; skip robot motion service")
    args = parser.parse_args()
    if config.robot != "b2" or not 0 < args.speed <= 1:
        parser.error("config.robot must be b2 and --speed must be in (0, 1]")
    if args.sim and (args.interface != "lo" or args.domain != 1):
        parser.error("--sim requires --interface lo --domain 1")
    if args.min_clearance < 0.15:
        parser.error("--min-clearance must be at least 0.15 m")
    if args.rest_foot_clearance is not None and args.rest_foot_clearance <= 0:
        parser.error("--rest-foot-clearance must be positive")
    if not args.sim and not args.preview and args.rest_foot_clearance is None:
        parser.error("hardware motion requires --rest-foot-clearance measured on the rig")

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    trajectory = make_trajectory(model, args.mode, args.seed, args.speed)
    collector_start_q = trajectory.sample(0.0).copy()
    if np.any(trajectory.positions < LOWER) or np.any(trajectory.positions > UPPER):
        raise ValueError("Trajectory exceeds B2 joint limits")
    names = [f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
    model_foot_bottom = float(foot_bottoms(model, trajectory.positions).min())
    print(f"Lowest modeled foot bottom: {model_foot_bottom:.3f} m relative to base origin", flush=True)
    if args.rig_check:
        rig_target, planned_bottom, target_bottom = lowest_foot_poses(model, trajectory)
        print(f"Rig check for planned {args.mode} (speed {args.speed}, seed {args.seed})", flush=True)
        print("Lowest-foot target, FL/FR/RL/RR hip/thigh/calf [rad]:", rig_target, flush=True)
        print(f"Foot bottom relative to base origin: planned {planned_bottom:.3f} m; rig target {target_bottom:.3f} m", flush=True)
        print(f"Motion: at most {RIG_CHECK_MAX_SPEED:.2f} rad/s, hold {RIG_CHECK_HOLD:.1f} s, then return to start", flush=True)
    else:
        print(trajectory.describe(names), flush=True)
    print(f"Gains: kp={config.Kp.tolist()}, kd={config.Kd.tolist()}", flush=True)
    print(f"Recording {RECORD_HZ} Hz; commands {COMMAND_HZ} Hz; stop if state age > {STATE_TIMEOUT}s", flush=True)
    if args.preview:
        return
    output_path = next_path(f"rig_check_{trajectory.name}" if args.rig_check else trajectory.name, args.sim)

    # Importing and constructing the subscriber does not create a command channel.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    state_lock = threading.Lock()
    state = {"timestamp": 0.0, "q": None, "dq": None, "fault": None}

    def on_state(msg: LowState_) -> None:
        if len(msg.motor_state) < 12:
            with state_lock:
                state["fault"] = "incomplete LowState"
            return
        q = np.array([m.q for m in msg.motor_state[:12]], dtype=float)[SDK_TO_DATASET]
        dq = np.array([m.dq for m in msg.motor_state[:12]], dtype=float)[SDK_TO_DATASET]
        with state_lock:
            if not np.isfinite(q).all() or not np.isfinite(dq).all():
                state["fault"] = "non-finite LowState"
            elif np.any(np.abs(dq) > 25.0):
                state["fault"] = "joint velocity exceeded 25 rad/s"
            else:
                state.update(timestamp=time.monotonic(), q=q, dq=dq)

    def snapshot():
        with state_lock:
            fault = state["fault"]
            age = time.monotonic() - state["timestamp"]
            q = None if state["q"] is None else state["q"].copy()
            dq = None if state["dq"] is None else state["dq"].copy()
        if fault:
            raise RuntimeError(fault)
        if q is None or age > STATE_TIMEOUT:
            raise RuntimeError(f"LowState stale ({age:.3f} s)")
        return q, dq

    ChannelFactoryInitialize(args.domain, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_state, 10)
    publisher = None
    stop = threading.Event()
    command_lock = threading.Lock()
    target = {"q": None, "fault": None}
    worker = None
    abort_worker = None
    q_rows, dq_rows, target_rows = [], [], []
    completed = False
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                start_q, _ = snapshot()
                break
            except RuntimeError:
                time.sleep(0.02)
        else:
            raise RuntimeError("No fresh B2 state within 5 seconds")
        if np.any(start_q < LOWER - 0.05) or np.any(start_q > UPPER + 0.05):
            raise RuntimeError("Current B2 posture lies outside the model joint limits (0.05 rad tolerance)")
        if args.rig_check:
            trajectory = rig_check_trajectory(collector_start_q, rig_target, trajectory)
            _, rig_path_bottom, _ = lowest_foot_poses(model, trajectory)
            print(f"Lowest model foot bottom during rig check: {rig_path_bottom:.3f} m relative to base origin", flush=True)
            print(trajectory.describe(names), flush=True)
        ramp = trajectories.reset_ramp_trajectory(
            start_q, collector_start_q, COMMAND_HZ,
            max_velocity=RIG_CHECK_MAX_SPEED,
        )
        if args.rest_foot_clearance is not None:
            start_bottom = foot_bottoms(model, start_q[None, :])[0]
            motion_bottom = foot_bottoms(model, np.concatenate((ramp.positions, trajectory.positions))).min(axis=0)
            max_descent = float(np.max(start_bottom - motion_bottom))
            predicted_gap = args.rest_foot_clearance - max_descent
            print(f"Measured rest foot gap {args.rest_foot_clearance:.3f} m; model maximum additional descent "
                  f"{max_descent:.3f} m; predicted minimum gap {predicted_gap:.3f} m", flush=True)
            if predicted_gap < args.min_clearance:
                raise RuntimeError(f"Predicted foot clearance below {args.min_clearance:.3f} m; no command publisher created")
        print("Fresh B2 state received. Current FL/FR/RL/RR joint positions:", start_q, flush=True)
        if args.sim:
            print("Simulation on loopback DDS domain 1.", flush=True)
        else:
            print("The robot must be securely fixed with feet clear. ReleaseMode may remove its active hold.", flush=True)
        if input("Type REST B2 to ramp to the collector start pose in simulation: " if args.sim else
                 "Type REST B2 to ramp to the collector start pose: ").strip() != "REST B2":
            print("Cancelled; no low-level command publisher created.")
            return
        _, start_dq = snapshot()
        if np.max(np.abs(start_dq)) > 0.5:
            raise RuntimeError("Robot is moving; wait for a stable posture before arming")

        if not args.sim:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            switcher = MotionSwitcherClient()
            switcher.SetTimeout(5.0)
            switcher.Init()
            status, current_mode = switcher.CheckMode()
            if status != 0:
                raise RuntimeError(f"CheckMode failed: {status}")
            print(f"Current motion mode: {current_mode}", flush=True)
            if current_mode.get("name"):
                if input("Type RELEASE B2 to release this mode: ").strip() != "RELEASE B2":
                    print("Cancelled; no command publisher created.")
                    return
                status, _ = switcher.ReleaseMode()
                if status != 0:
                    raise RuntimeError(f"ReleaseMode failed: {status}")

        publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        publisher.Init()
        command = unitree_go_msg_dds__LowCmd_()
        command.head[0], command.head[1] = 0xFE, 0xEF
        command.level_flag = 0xFF
        command.gpio = 0
        for motor in command.motor_cmd:
            motor.mode = 0x01  # Official Python B2 example; C++ example differs.
            motor.q, motor.dq, motor.kp, motor.kd, motor.tau = 0.0, 0.0, 0.0, 0.0, 0.0
        crc = CRC()
        with command_lock:
            target["q"] = start_q.copy()

        def publish_loop():
            next_tick = time.monotonic()
            while not stop.is_set():
                try:
                    actual_q, actual_dq = snapshot()
                    with command_lock:
                        desired = target["q"].copy()
                    if args.rig_check:
                        measured_speed = float(np.max(np.abs(actual_dq)))
                        if measured_speed > RIG_CHECK_MAX_MEASURED_SPEED:
                            raise RuntimeError(
                                f"Rig check measured joint speed {measured_speed:.3f} rad/s exceeded "
                                f"{RIG_CHECK_MAX_MEASURED_SPEED:.1f} rad/s"
                            )
                        if np.max(np.abs(actual_q - desired)) > RIG_CHECK_MAX_TRACKING_ERROR:
                            raise RuntimeError("Rig check tracking error exceeded 0.3 rad")
                    sdk_target = desired[DATASET_TO_SDK]
                    for i in range(12):
                        command.motor_cmd[i].q = float(sdk_target[i])
                        command.motor_cmd[i].kp = float(config.Kp[DATASET_TO_SDK[i]])
                        command.motor_cmd[i].kd = float(config.Kd[DATASET_TO_SDK[i]])
                    command.crc = crc.Crc(command)
                    if stop.is_set():
                        break
                    publisher.Write(command)
                except Exception as exc:
                    with command_lock:
                        target["fault"] = str(exc)
                    stop.set()
                    break
                next_tick += 1.0 / COMMAND_HZ
                time.sleep(max(0.0, next_tick - time.monotonic()))

        worker = threading.Thread(target=publish_loop, daemon=True)
        worker.start()

        def set_target(q):
            if stop.is_set():
                raise RuntimeError(target["fault"] or "Command publication stopped")
            with command_lock:
                target["q"] = np.asarray(q, dtype=float).copy()

        ramp_start = time.monotonic()
        while time.monotonic() - ramp_start < ramp.duration:
            set_target(ramp.sample(time.monotonic() - ramp_start))
            time.sleep(1.0 / RECORD_HZ)
        set_target(trajectory.sample(0.0))
        print("Collector start pose reached and held.", flush=True)

        arm_prompt = "Type ARM B2 to start the trajectory (or anything else to cancel): "
        if sys.stdin.isatty():
            print(arm_prompt, end="", flush=True)
            while not stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    arm_response = sys.stdin.readline().strip()
                    break
            else:
                raise RuntimeError(target["fault"] or "Command publication stopped before trajectory arm")
        else:
            arm_response = input(arm_prompt).strip()
        if arm_response != "ARM B2":
            print("Trajectory cancelled; low-level commands will stop.", flush=True)
            return

        if sys.stdin.isatty():
            def abort_on_enter():
                while not stop.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not ready:
                        continue
                    if sys.stdin.readline() == "":
                        return
                    with command_lock:
                        target["fault"] = "Operator abort requested"
                    stop.set()
                    return

            print("Trajectory armed. Press Enter to abort and stop low-level commands.", flush=True)
            abort_worker = threading.Thread(target=abort_on_enter, daemon=True)
            abort_worker.start()

        print("Starting trajectory recording.", flush=True)

        start = time.monotonic()
        for i in range(trajectory.positions.shape[0]):
            due = start + i / RECORD_HZ
            time.sleep(max(0.0, due - time.monotonic()))
            desired = trajectory.positions[i]
            set_target(desired)
            q, dq = snapshot()
            q_rows.append(q)
            dq_rows.append(dq)
            target_rows.append(desired.copy())
        set_target(trajectory.sample(trajectory.duration))
        completed = True
        print("Trajectory complete; stopping command publication.", flush=True)
    finally:
        stop.set()
        if abort_worker is not None:
            abort_worker.join(timeout=0.2)
        if worker is not None:
            worker.join(timeout=1.0)
        if publisher is not None:
            publisher.Close()
        subscriber.Close()
        if len(q_rows) >= 2:
            path = output_path if completed else next_path(f"aborted_{trajectory.name}", args.sim)
            np.savez(path, time=np.arange(len(q_rows)) / RECORD_HZ,
                     dof_pos=np.asarray(q_rows), dof_vel=np.asarray(dq_rows),
                     des_dof_pos=np.asarray(target_rows),
                     des_dof_vel=np.zeros((len(q_rows), 12)),
                     kp=np.asarray(config.Kp), kd=np.asarray(config.Kd), **trajectory.metadata)
            print(f"Saved {'complete' if completed else 'partial'} recording to {path}", flush=True)


if __name__ == "__main__":
    main()
