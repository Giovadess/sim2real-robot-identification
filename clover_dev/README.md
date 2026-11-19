# clover_dev: B2 models and actuator parameters

Files for using the identified B2 model in training (Isaac Lab / Isaac Sim) and in MuJoCo.

| File | What it is |
|---|---|
| `b2_urdf_masses.xml` | MuJoCo B2 (floating base) with the leg masses, CoM and inertias of `b2z1.urdf`; joint armature/damping/friction at the nominal defaults (0.1 / 1 / 0). |
| `b2_urdf_masses_fitted.xml` | Same model, with per-joint armature, damping and frictionloss from the MuJoCo fit done **with these URDF leg masses**. |
| `isaaclab_identified_parameters.py` | Per-joint armature, viscous friction and dynamic friction from the **Isaac Lab** fit. |
| `b2_pace_actuators.py` | `PaceDCMotorCfg` blocks for hip, thigh and calf with those Isaac Lab values and the gamma-zero deployment gains. |

All fits use the same data: 20 real sideways recordings on the fixed-base rig at kp 100 / kd 5, with `sideways_14` held out. Joint order in the lists: FL, FR, RL, RR, each hip, thigh, calf.

## Which files go together

The joint parameters were identified together with specific leg masses. Keep each pair together.

| Leg masses | Joint parameters | Use |
|---|---|---|
| MJCF legs (thigh 7.46 kg) | `isaaclab_identified_parameters.py` / `b2_pace_actuators.py` | Isaac Lab / Isaac Sim training. Robot model: `../robot_model/b2/b2z1_mjcf_legs.urdf` (b2z1 with these legs) or basic-locomotion-isaaclab's `b2_asset/b2.usd` |
| URDF legs (thigh 4.81 kg) | `b2_urdf_masses_fitted.xml` | MuJoCo only. There is no Isaac Lab fit with these masses. |

The MJCF legs fit the real data better (held-out mean joint RMSE 0.0103 rad vs 0.0118 rad with the URDF legs; thighs 2-3x worse with the URDF legs). With the URDF legs the fit compensates at the hips (armature about 0.45 kg m², damping at its lower bound), so treat `b2_urdf_masses_fitted.xml` as a comparison model, not the reference.

## Isaac Lab usage

```python
from b2_pace_actuators import B2_HIP_ACTUATOR_CFG, B2_THIGH_ACTUATOR_CFG, B2_CALF_ACTUATOR_CFG
actuators={"hip": B2_HIP_ACTUATOR_CFG, "thigh": B2_THIGH_ACTUATOR_CFG, "calf": B2_CALF_ACTUATOR_CFG}
```

- Gains: hip/thigh kp 80 / kd 2, calf kp 405 / kd 10.125 (gamma-zero bundle `e274_6.5B`). The policy must be trained and deployed with the same gains.
- Armature, friction and viscous damping are applied by the actuator config at runtime. A URDF cannot hold armature, so these values are not in the URDF.
- Delay is randomized over 0-2 simulation steps (`max_delay=2`); the fit could not pin delay down and the stiff calf gains make it matter.
- Viscous damping is weakly identified (about the size of kd on hip and thigh); randomizing it (for example ±50%) is advisable.
- Validated only for slow motion at kp 100 / kd 5; motor mode 0x01.
