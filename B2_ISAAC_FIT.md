# B2 PACE-style system identification in Isaac Lab

Status 2026-11-18: task written, **not yet run** (no Isaac Lab environment on the laptop, see Blockers).

## What was added
- `sysid_isaaclab/tasks/b2_pace_env_cfg.py`: task `IsaacLab-Pace-b2` (go2 layout: 49 parameters = armature, damping, friction, bias, delay).
- `sysid_isaaclab/tasks/register_my_tasks.py`: registers it.
- No dataset adapter is needed: `datasets/b2/sideways_*.pt` already has `dof_pos` / `des_dof_pos` (T, 12) in FL, FR, RL, RR order and 12-vector `kp` / `kd`, which `my_fit.py` reads as is (extra keys such as `excitation_seed` are ignored).

## Data used
- Fit: `sideways_1..21.pt` except `sideways_14.pt` (20 recordings, 24 s, 200 Hz = the task `sim.dt`).
- Held out for verification: `sideways_14.pt` (`excitation_seed` = 14). Seed to file mapping is not the file number in general (e.g. `sideways_1` is seed 12); 14 happens to match.
- Each recording gets its own group of environments, so candidates per iteration = `num_envs // 20`.

## Model assets
- `robot_model/b2/generated_usd/b2.usda` (committed, MuJoCo USD Converter output). 12 revolute joints named `{FL,FR,RL,RR}_{hip,thigh,calf}_joint`.

## Command (needs the pixi `isaaclab` environment)
```bash
cd ~/mujoco_dev/sim2real-robot-identification   # config.py must have robot = 'b2'
pixi run -e isaaclab python sysid_isaaclab/my_fit.py --num_envs 4096        # CMA-ES
# inertial parameters: add e.g. --identify-link-mass physics=physx
```
`num_envs` should be a multiple of 20. `labserver` is the better home for 8192+ envs; paths there differ.

## Assumptions to check
- Stiffness 100 / damping 5 (the saved kp/kd). The task has one value for all joints; `my_fit.py` rejects recordings with different gains.
- Effort limits 200 / 200 / 300 Nm from `b2.xml` ctrlrange.
- **Velocity limits 23 / 23 / 14 rad/s are assumed**, not found in the repo; replace with datasheet values.
- Bounds: armature 1e-5–1, damping 0–10, friction 0–10, bias fixed 0, delay 0–5 steps. Wider than go2 because B2 is larger; not tuned.
- Base is fixed by the PACE env as for the other robots; the rig (belly support) is not modelled.
- The USD joint drive/armature (`mjc:armature = 0.1`) is overwritten by the optimizer parameters.

## Blockers
- `pixi run -e isaaclab` environment not installed on this laptop (`.pixi/envs` missing; Isaac Lab 3.0.0 plus torch from NVIDIA/PyTorch indexes). Install with `pixi install -e isaaclab`.
- Nothing here was executed in Isaac Lab; first run should check that the first-iteration loss is sensible and that `sideways_14` validation (replay with the best parameters) is done separately.
