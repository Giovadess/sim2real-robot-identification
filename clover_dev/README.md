# clover_dev: identified Unitree B2 model for training and simulation

This folder holds the result of the Unitree B2 actuator identification (system identification, "sysid") in the
`sim2real-robot-identification` repository, packaged for robot-learning use: Isaac Lab / Isaac Sim training and
MuJoCo simulation. It is written to be read by people and by coding agents; every fact below is explicit.

The method is PACE (Bjelonic, Tischhauser and Hutter, IJRR 2026; see [Reference](#reference)). A copy of the
paper is in this folder: `PACE_Bjelonic2026_IJRR.pdf`.

## Quick facts

| Key | Value |
|---|---|
| Robot | Unitree B2 quadruped, 12 actuated joints, motor mode 0x01 |
| Joint order (all lists in this folder) | FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf |
| Joint names in models | `{FL,FR,RL,RR}_{hip,thigh,calf}_joint` |
| Identified per joint | armature [kg m²], viscous damping [N m s/rad], dry (Coulomb) friction [N m] |
| Also in the Isaac Lab fit | encoder bias (fixed at 0), actuation delay (0.61 sim steps, truncated to 0 by the simulator) |
| Not identified | link masses, CoM, inertias (taken from the robot model) |
| Data | 21 real recordings, "sideways" excitation, 24 s each at 200 Hz, base fixed on a rig, feet clear |
| Gains during data collection | kp 100, kd 5 on all joints |
| Fit set / held-out set | 20 recordings / `sideways_14` |
| Validated range | slow motion (peak joint speed 0.89 rad/s), fixed base, kp 100 / kd 5 |
| Held-out result (Isaac Lab) | RMS joint error 0.0087 rad fitted vs 0.0377 rad nominal |

## Files

| Path (relative to this folder) | Format | Contents |
|---|---|---|
| `isaaclab_identified_parameters.py` | Python constants | Isaac Lab fit: `armature`, `viscous_friction`, `dynamic_friction`, `bias` (12 values each, joint order above), `delay`, `Kp`, `Kd` (collection gains). |
| `b2_pace_actuators.py` | Python, Isaac Lab | `B2_HIP_ACTUATOR_CFG`, `B2_THIGH_ACTUATOR_CFG`, `B2_CALF_ACTUATOR_CFG`: `PaceDCMotorCfg` (from `basic_locomotion_isaaclab.actuators`) with the Isaac Lab fit values and the gamma-zero deployment gains. |
| `b2_urdf_masses.xml` | MJCF (MuJoCo) | B2, floating base, leg inertials from `b2z1.urdf`; joints at nominal defaults (armature 0.1, damping 1, frictionloss 0). Meshes load from `../robot_model/b2/assets`. |
| `b2_urdf_masses_fitted.xml` | MJCF (MuJoCo) | Same model with per-joint `armature`, `damping`, `frictionloss` from the MuJoCo fit done with the `b2z1.urdf` leg masses. |
| `../robot_model/b2/b2z1_fitting_test.urdf` | URDF | `b2z1.urdf` (B2 + Z1 arm) with the leg inertials replaced by the MJCF legs the Isaac Lab fit used. For Isaac Sim import. |
| `PACE_Bjelonic2026_IJRR.pdf` | PDF | The PACE paper. |

## Which files go together (rules)

Joint parameters were identified together with specific leg masses. Use them only with the matching model.

| Leg model | Leg masses (hip / thigh / calf / foot, kg) | Matching joint parameters | Intended use |
|---|---|---|---|
| MJCF legs (reference) | 2.53 / 7.46 / 0.68 / 0.02 (leg 10.68) | `isaaclab_identified_parameters.py`, `b2_pace_actuators.py` | Isaac Lab / Isaac Sim training, with `../robot_model/b2/b2z1_fitting_test.urdf` or basic-locomotion-isaaclab's `b2_asset/b2.usd` |
| `b2z1.urdf` legs | 2.95 / 4.81 / 0.40 / 0.13 (leg 8.29, rotors included) | `b2_urdf_masses_fitted.xml` only | MuJoCo comparison model |

- MUST: pair `b2_pace_actuators.py` with a robot model that has the MJCF leg masses.
- MUST NOT: use `b2_pace_actuators.py` / `isaaclab_identified_parameters.py` with the original `b2z1.urdf` leg masses. No Isaac Lab fit exists for those masses.
- MUST: train and deploy a policy with the same PD gains.
- Prefer the MJCF-leg pairing. It fits the real data better: held-out mean joint RMSE 0.0103 rad vs 0.0118 rad (MuJoCo replay), thighs 2-3x worse with the URDF legs; with the URDF legs the fit compensates at the hips (armature about 0.45 kg m², damping at its lower bound 0.01).

## The URDF (`../robot_model/b2/b2z1_fitting_test.urdf`)

- Changed from `b2z1.urdf`: only the `<inertial>` of the 16 leg links (`{FL,FR,RL,RR}_{hip,thigh,calf,foot}`), set to the MJCF body mass, CoM and full inertia tensor in the URDF link frame; the 12 `*_rotor` links set to 1e-6 kg (the MJCF has no separate rotors; their inertia about the joint axis is represented by the fitted armature).
- Unchanged: base, Z1 arm, joint limits, meshes, link/joint structure (49 links, 48 joints). Mesh paths are relative (`../meshes/...`): place the file next to `b2z1.urdf`.
- No `<dynamics>` entries: URDF has no armature field, and the actuator config applies armature, friction and damping at runtime (overwriting importer values).
- Verified in MuJoCo: leg joint-space inertia and gravity torque match the MJCF to about 1e-9 at six poses. Check after Isaac Sim import: each leg sums to about 10.7 kg.

## Isaac Lab usage

```python
from b2_pace_actuators import B2_HIP_ACTUATOR_CFG, B2_THIGH_ACTUATOR_CFG, B2_CALF_ACTUATOR_CFG

actuators = {"hip": B2_HIP_ACTUATOR_CFG, "thigh": B2_THIGH_ACTUATOR_CFG, "calf": B2_CALF_ACTUATOR_CFG}
```

| Group | kp [N m/rad] | kd [N m s/rad] | Effort limit [N m] | Velocity limit [rad/s] | max_delay [sim steps] |
|---|---|---|---|---|---|
| hip | 80 | 2 | 200 | 23 | 2 |
| thigh | 80 | 2 | 200 | 23 | 2 |
| calf | 405 | 10.125 | 300 | 14 | 2 |

- Gains are the gamma-zero B2 deployment gains (bundle `e274_6.5B` in `~/gamma_zero_tienkgun/deployment/unitree_b2`). Change them in `b2_pace_actuators.py` if the deployment uses others.
- The identified values are joint properties and are used unchanged at these gains, but they were validated only at kp 100 / kd 5.
- Delay is randomized over 0-2 steps because the slow data could not pin it down and the stiff calf gains make it matter.
- Viscous damping is weakly identified (per-joint values differ between the Isaac Lab and MuJoCo fits; on hip and thigh it is about the size of kd 2). Randomize it, for example ±50%.

## MuJoCo usage

```python
import mujoco
model = mujoco.MjModel.from_xml_path("clover_dev/b2_urdf_masses_fitted.xml")
```

The repository's own reference MuJoCo fit (MJCF legs) is `../sysid_mujoco/generated/b2/b2_fitted.xml` (fixed base). MuJoCo and Isaac Lab fit values are separate sets; use each only in its own simulator.

## Provenance

| Item | Location in the repository |
|---|---|
| Recordings | `datasets/b2/sideways_{1..21}.pt` |
| Isaac Lab task | `sysid_isaaclab/tasks/b2_pace_env_cfg.py` (`IsaacLab-Pace-b2`), fit with `sysid_isaaclab/my_fit.py --num_envs 4096`, CMA-ES, 144 iterations |
| Held-out replay | `sysid_isaaclab/replay_heldout.py` |
| MuJoCo fit | `sysid_mujoco/my_fit.py` (MuJoCo sysid toolbox) |
| Notes | `B2_ISAAC_FIT.md`, `B2_SYSID_STATUS.md` |

## Reference

F. Bjelonic, F. Tischhauser, and M. Hutter, "Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse
Legged Robots," The International Journal of Robotics Research, 2026. doi:10.1177/02783649261459628.
arXiv:2509.06342. Local copy: `PACE_Bjelonic2026_IJRR.pdf`. Code: `sysid_isaaclab/pace-sim2real`, https://pace.filipbjelonic.com

```bibtex
@article{bjelonic2025towards,
  title   = {Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots},
  author  = {Bjelonic, Filip and Tischhauser, Fabian and Hutter, Marco},
  journal = {The International Journal of Robotics Research},
  year    = {2026},
  doi     = {10.1177/02783649261459628}
}
```
