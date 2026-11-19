# Unitree B2 actuator configs for Isaac Lab (basic-locomotion-isaaclab), PaceDCMotorCfg as in go2_asset.py.
#
# Joint parameters: Isaac Lab PACE fit (isaaclab_identified_parameters.py in this folder), identified on the
# fixed-base rig at kp 100 / kd 5 with the MJCF leg masses (thigh 7.46 kg). They are joint properties, so they
# are used unchanged at other gains. Pair them with a robot model that has the same leg masses
# (robot_model/b2/b2z1_mjcf_legs.urdf or basic-locomotion-isaaclab's b2_asset/b2.usd), not with the
# b2z1.urdf leg masses.
#
# Gains: gamma-zero B2 deployment gains (bundle e274_6.5B): hip/thigh kp 80 kd 2, calf kp 405 kd 10.125.
# Train and deploy with the same gains.
#
# Order of the lists: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf,
#                     RR_hip, RR_thigh, RR_calf

from basic_locomotion_isaaclab.actuators import PaceDCMotorCfg

armature = [0.09274006634950638, 0.17894206941127777, 0.3847358226776123, 0.12008873373270035, 0.1410508155822754, 0.44124636054039, 0.08949188888072968, 0.15425632894039154, 0.47201013565063477, 0.04641764983534813, 0.15626436471939087, 0.364768922328949]
viscous_friction = [3.6811976432800293, 1.5729182958602905, 1.082211971282959, 4.246136665344238, 2.7687530517578125, 1.0465824604034424, 1.4282200336456299, 0.8275148272514343, 1.0474765300750732, 3.928321599960327, 3.6634039878845215, 1.6373159885406494]
dynamic_friction = [3.117905616760254, 2.9382073879241943, 7.32740592956543, 2.979374408721924, 2.349820613861084, 7.877301216125488, 3.0709962844848633, 3.010183572769165, 8.40671443939209, 2.510951519012451, 2.911743640899658, 6.202816963195801]
bias = [0.0] * 12  # encoder bias was fixed to 0 in the fit

LEGS = ("FL", "FR", "RL", "RR")


def _per_joint(values, joint, offset):
    """{'FL_<joint>_joint': values[offset], 'FR_...': values[offset + 3], ...}"""
    return {f"{leg}_{joint}_joint": values[offset + 3 * i] for i, leg in enumerate(LEGS)}


def _b2_actuator(joint, offset, kp, kd, effort, velocity):
    return PaceDCMotorCfg(
        joint_names_expr=[f".*_{joint}_joint"],
        saturation_effort=effort,
        effort_limit=effort,
        velocity_limit=velocity,
        stiffness={".*": kp},  # P gain [N m/rad]
        damping={".*": kd},  # D gain [N m s/rad]
        encoder_bias=_per_joint(bias, joint, offset),  # [rad]
        # coulomb friction: friction = dynamic_friction, as in go2_asset.py
        friction=_per_joint(dynamic_friction, joint, offset),  # static friction [N m]
        dynamic_friction=_per_joint(dynamic_friction, joint, offset),  # dynamic friction [N m]
        viscous_friction=_per_joint(viscous_friction, joint, offset),  # viscous friction [N m s/rad]
        armature=_per_joint(armature, joint, offset),  # reflected rotor inertia [kg m^2]
        # the fit found ~0.6 steps (truncated to 0); delay is weakly identified and matters more at the stiff
        # calf gains, so it is randomized over 0..2 simulation steps
        max_delay=2,
    )


# effort limits as used in the identification model (b2.xml ctrlrange); velocity limits from the B2 URDF
B2_HIP_ACTUATOR_CFG = _b2_actuator("hip", 0, kp=80.0, kd=2.0, effort=200.0, velocity=23.0)
B2_THIGH_ACTUATOR_CFG = _b2_actuator("thigh", 1, kp=80.0, kd=2.0, effort=200.0, velocity=23.0)
B2_CALF_ACTUATOR_CFG = _b2_actuator("calf", 2, kp=405.0, kd=10.125, effort=300.0, velocity=14.0)

# usage in the ArticulationCfg:
#   actuators={"hip": B2_HIP_ACTUATOR_CFG, "thigh": B2_THIGH_ACTUATOR_CFG, "calf": B2_CALF_ACTUATOR_CFG}
