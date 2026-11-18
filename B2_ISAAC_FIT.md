# B2 PACE-style system identification in Isaac Lab

Status 2026-11-19: **fit done and validated on the held-out recording** (see Results). Fitted parameters: `robot_model/b2/isaaclab_identified_parameters.py`.

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
pixi run -e isaaclab python sysid_isaaclab/my_fit.py --num_envs 4096        # CMA-ES (no --headless flag in Isaac Lab 3.0)
# inertial parameters: add e.g. --identify-link-mass physics=physx
```
`num_envs` should be a multiple of 20 (4096 runs 4080 envs = 204 candidates x 20 recordings). `labserver` is the better home for 8192+ envs; paths there differ.

Validation on a recording that was not fitted (fixed parameters, no optimization):
```bash
pixi run -e isaaclab python sysid_isaaclab/replay_heldout.py \
    --params sysid_isaaclab/pace-sim2real/logs/pace/b2_sim/<run>/mean_143.pt \
    --dataset datasets/b2/sideways_14.pt datasets/b2/sideways_1.pt --baseline
```

## Assumptions to check
- Stiffness 100 / damping 5 (the saved kp/kd). The task has one value for all joints; `my_fit.py` rejects recordings with different gains.
- Effort limits 200 / 200 / 300 Nm from `b2.xml` ctrlrange.
- **Velocity limits 23 / 23 / 14 rad/s are assumed**, not found in the repo; replace with datasheet values.
- Bounds: armature 1e-5–1, damping 0–10, friction 0–10, bias fixed 0, delay 0–5 steps. Wider than go2 because B2 is larger; not tuned.
- Base is fixed by the PACE env as for the other robots; the rig (belly support) is not modelled.
- The USD joint drive/armature (`mjc:armature = 0.1`) is overwritten by the optimizer parameters.

## Results (run `26_11_18_22-29-28`, laptop RTX 5060, 4096 envs, ~21 s/iteration)
- CMA-ES stopped at iteration 144 (max 200, the `epsilon` stop criterion). Best candidate score 0.00283 (iteration 0) -> 0.00072; flat after about iteration 100.
- Logs/checkpoints: `sysid_isaaclab/pace-sim2real/logs/pace/b2_sim/26_11_18_22-29-28/` (not committed); `mean_143.pt` = 49 sim params of the CMA-ES mean (armature 12, damping 12, friction 12, bias 12, delay 1).
- Held-out replay of `mean_143.pt` (`replay_heldout.py`, rms of sim - real joint position over all joints, rad):

| recording | fitted | nominal (armature 0.1, damping 1, friction 0, delay 0) |
|---|---|---|
| `sideways_14` (held out) | 0.0087 (max 0.031) | 0.0377 |
| `sideways_1` (fit) | 0.0076 | 0.0385 |
| `sideways_12` (fit) | 0.0080 | 0.0381 |

  Held-out error is about 8-14% above the training error and 4.3x below the nominal model. The replay reproduces the fit score (0.0007-0.0008 on training files), so it uses the same metric as the fit.
- Parameter ranges: armature 0.05-0.47 (hip < thigh < calf), friction 2.3-3.1 (hip, thigh) and 6.2-8.4 (calf), damping 0.8-4.2, delay 0.61 steps.
- The simulator truncates the delay to an integer, so the effective delay is **0 steps**; the improvement comes from armature, friction and damping. A 0-5 step bound is wider than needed.
- Comparison with the MuJoCo fit (`sysid_mujoco/results/b2/final_all21`): per-joint armature and friction agree (correlation 0.97 and 0.99, same means 0.23 and about 4.3). **Damping does not agree** (correlation 0.09; calf damping 1.1-1.7 here vs 1.3-7.0 in MuJoCo, where RL calf sits at its bound of 7). Dry friction and damping are partly interchangeable at this speed; it is not known whether go2 shows the same. MuJoCo's calf friction is capped at its bound of 7.0 on three joints; here the bound of 10 is not reached.

## Open items
- Velocity limits (23 / 23 / 14 rad/s) are still assumed: `B2+User+Manual.pdf` is a user guide and has no joint specs. They do not matter for these recordings (peak joint speed about 1 rad/s), only for faster motions.
- Only one recording was held out, and all data are slow (speed scale 0.3), fixed base on the rig.
