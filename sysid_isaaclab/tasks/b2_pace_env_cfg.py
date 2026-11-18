# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.assets import ArticulationCfg
from .pace_actuator import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
from .physics_cfg import PacePhysicsCfg
import torch

import os
import toml
ISAAC_ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

# Real B2 recordings: datasets/b2/sideways_1..21.pt. Seed 14 (sideways_14) is held out for validation.
B2_HELD_OUT_FILES = ("sideways_14.pt",)
B2_FIT_FILES = [
    f"{ISAAC_ASSET_DIR}/../../datasets/b2/sideways_{i}.pt"
    for i in range(1, 22)
    if f"sideways_{i}.pt" not in B2_HELD_OUT_FILES
]

# effort limits are the ctrlrange of robot_model/b2/b2.xml
B2_HIP_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_joint"],
    saturation_effort=200.0,
    effort_limit=200.0,
    velocity_limit=23.0,  # ASSUMED, not in the repo: check the B2 datasheet
    stiffness={".*": 100.0},  # kp of the real recordings (Nm/rad)
    damping={".*": 5.0},  # kd of the real recordings (Nm s/rad)
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=5,  # max delay in simulation steps
)

B2_THIGH_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_thigh_joint"],
    saturation_effort=200.0,
    effort_limit=200.0,
    velocity_limit=23.0,  # ASSUMED, not in the repo: check the B2 datasheet
    stiffness={".*": 100.0},  # kp of the real recordings (Nm/rad)
    damping={".*": 5.0},  # kd of the real recordings (Nm s/rad)
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=5,  # max delay in simulation steps
)

B2_CALF_ACTUATOR_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_calf_joint"],
    saturation_effort=300.0,
    effort_limit=300.0,
    velocity_limit=14.0,  # ASSUMED, not in the repo: check the B2 datasheet
    stiffness={".*": 100.0},  # kp of the real recordings (Nm/rad)
    damping={".*": 5.0},  # kd of the real recordings (Nm s/rad)
    encoder_bias={".*": 0.0},
    friction={".*": 0.0},
    dynamic_friction={".*": 0.0},
    viscous_friction={".*": 0.0},
    max_delay=5,  # max delay in simulation steps
)

B2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/../../robot_model/b2/generated_usd/b2.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*L_hip_joint": 0.,
            ".*R_hip_joint": 0.,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.8,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"hip": B2_HIP_ACTUATOR_CFG, "thigh": B2_THIGH_ACTUATOR_CFG, "calf": B2_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)


@configclass
class B2PaceCfg(PaceCfg):
    """Pace configuration for B2 robot."""
    robot_name: str = "b2_sim"
    data_dir: list[str] = B2_FIT_FILES  # all real recordings except the held-out one
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    joint_order: list[str] = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:12, 0] = 1e-5
        self.bounds_params[:12, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[12:24, 1] = 10.0  # dof_damping between 0.0 - 10.0 [Nm s/rad]
        self.bounds_params[24:36, 1] = 10.0  # friction between 0.0 - 10.0 [Nm]
        self.bounds_params[36:48, 0] = -0.0
        self.bounds_params[36:48, 1] = 0.0  # bias fixed to 0 (as go2/a2)
        self.bounds_params[48, 1] = 5.0  # delay between 0.0 - 5.0 [sim steps]


@configclass
class B2PaceSceneCfg(PaceSim2realSceneCfg):
    """Configuration for B2 robot in Pace Sim2Real environment."""
    robot: ArticulationCfg = B2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class B2PaceEnvCfg(PaceSim2realEnvCfg):

    scene: B2PaceSceneCfg = B2PaceSceneCfg()
    sim2real: PaceCfg = B2PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.005  # 200Hz simulation
        self.sim.physics = PacePhysicsCfg()  # physics backend presets, same as basic-locomotion-isaaclab
        self.sim.use_newton_actuators = False  # PaceDCMotorCfg is not a Newton-native actuator
        self.decimation = 1  # 200Hz control
