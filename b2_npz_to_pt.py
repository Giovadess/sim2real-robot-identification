"""Convert a native B2 SDK2 recording to the existing Torch SysID dataset format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


ARRAY_KEYS = ("time", "dof_pos", "dof_vel", "des_dof_pos", "des_dof_vel")


def convert(source: Path) -> Path:
    if source.suffix != ".npz":
        raise ValueError("Expected an .npz recording")
    destination = source.with_suffix(".pt")
    if destination.exists():
        raise FileExistsError(destination)
    with np.load(source, allow_pickle=False) as recording:
        data = {}
        for key in ARRAY_KEYS:
            array = np.asarray(recording[key], dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {key}")
            data[key] = torch.from_numpy(array.copy())
        steps = data["time"].numel()
        if steps < 2 or any(data[key].shape != (steps, 12) for key in ARRAY_KEYS[1:]):
            raise ValueError("Recording must have at least two 12-joint samples")
        for key in ("kp", "kd"):
            gains = np.asarray(recording[key], dtype=np.float64)
            if gains.shape != (12,) or not np.isfinite(gains).all():
                raise ValueError(f"Invalid {key}")
            data[key] = gains.copy()
        for key in recording.files:
            if key not in (*ARRAY_KEYS, "kp", "kd"):
                value = recording[key]
                if value.shape != ():
                    raise ValueError(f"Unexpected metadata shape: {key}")
                data[key] = value.item()
    torch.save(data, destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    print(convert(args.recording))
