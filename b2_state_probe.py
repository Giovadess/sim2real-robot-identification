"""Read B2 LowState through SDK2 without creating a command publisher."""

from __future__ import annotations

import argparse
import math
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="enp98s0")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.samples < 2 or args.timeout <= 0:
        parser.error("--samples must be at least 2 and --timeout must be positive")

    received: list[tuple[float, list[float], list[float]]] = []
    done = threading.Event()

    def on_state(msg: LowState_) -> None:
        if done.is_set() or len(msg.motor_state) < 12:
            return
        positions = [float(m.q) for m in msg.motor_state[:12]]
        velocities = [float(m.dq) for m in msg.motor_state[:12]]
        if not all(math.isfinite(v) for v in positions + velocities):
            return
        received.append((time.monotonic(), positions, velocities))
        if len(received) >= args.samples:
            done.set()

    ChannelFactoryInitialize(args.domain, args.interface)
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_state, 10)
    try:
        done.wait(args.timeout)
    finally:
        subscriber.Close()

    if not received:
        raise SystemExit("No finite 12-motor LowState received before timeout")
    if len(received) > 1 and received[-1][0] > received[0][0]:
        duration = received[-1][0] - received[0][0]
        print(f"Received {len(received)} states in {duration:.3f} s ({(len(received)-1)/duration:.1f} Hz)")
    else:
        print("Received 1 state")
    print("Expected SDK motor index order: FR, FL, RR, RL; each hip, thigh, calf")
    for index, (position, velocity) in enumerate(zip(received[-1][1], received[-1][2])):
        print(f"{index:2d}: q={position:+.4f} rad  dq={velocity:+.4f} rad/s")


if __name__ == "__main__":
    main()
