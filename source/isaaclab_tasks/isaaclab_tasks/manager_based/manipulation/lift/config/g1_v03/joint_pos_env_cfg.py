from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import G1_WITH_HAND_CFG  # isort: skip

import os.path as osp

import math
from isaaclab.sensors import TiledCameraCfg, CameraCfg
import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend
import isaaclab.sim as sim_utils


@configclass
class G1CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set g1 as robot
        self.scene.robot = G1_WITH_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [-0.6, 0.0, 0.0]
        
        # Set actions for the specific robot type (g1)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                # ".*_hip_.*_joint",
                # ".*_knee_joint",
                # ".*_ankle_.*_joint",
                # "waist_.*_joint",
                # ".*_shoulder_.*_joint",
                # ".*_elbow_joint",
                # ".*_wrist_.*_joint",

                "right_shoulder_.*",
                "right_elbow_joint",
                "right_wrist_.*",

                # ".*_index_.*_joint",
                # ".*_little_.*_joint",
                # ".*_middle_.*_joint",
                # ".*_ring_.*_joint",
                # ".*_thumb_.*_joint",
            ],
            scale=1.0,                  
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg( 
            asset_name="robot",
            joint_names=[                
                "right_index_.*_joint",
                "right_little_.*_joint",
                "right_middle_.*_joint",
                "right_ring_.*_joint",
                "right_thumb_.*_joint",
                ],
            # open_command_expr={    
            #     "right_index_.*_joint": 0.0,
            #     "right_little_.*_joint": 0.0,
            #     "right_middle_.*_joint" : 0.0,
            #     "right_ring_.*_joint" : 0.0,
            #     "right_thumb_.*_joint" : 0.0},
            # close_command_expr={    
            #     "right_index_.*_joint": 0.8,
            #     "right_little_.*_joint": 0.8,
            #     "right_middle_.*_joint" : 0.8,
            #     "right_ring_.*_joint" : 0.8,
            #     "right_thumb_.*_joint" : 0.8},
            open_command_expr={    
                "right_index_.*_joint": 0.3,
                "right_little_.*_joint": 0.3,
                "right_middle_.*_joint" : 0.3,
                "right_ring_.*_joint" : 0.3,
                "right_thumb_.*_joint" : 0.3},
            close_command_expr={    
                "right_index_.*_joint": 0.8,
                "right_little_.*_joint": 0.8,
                "right_middle_.*_joint" : 0.8,
                "right_ring_.*_joint" : 0.8,
                "right_thumb_.*_joint" : 0.8},

        )


        # Set the body name for the end effector
        self.commands.object_pose.body_name = "right_middle_1"
        self.commands.object_pose.ranges.roll = [-math.pi / 2, 0.0]


        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.0], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                # scale=(1.0,1.0,1.0),
                scale=(0.8,0.8,0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_middle_1",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=[-0.03, -0.04, -0.015],
                        pos=[-0.03, -0.03, -0.015],
                    ),
                ),
            ],
        )


@configclass
class G1CubeLiftEnvCfg_PLAY(G1CubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.scene.camera_sensor = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",
            update_period=0.0,
            height=1080,
            width=1920,
            data_types=["rgb", "distance_to_camera"],
            # spawn = sim_utils.PinholeCameraCfg(
            #     focal_length=0.193,  # cm
            #     focus_distance=400.0,  # m
            #     horizontal_aperture=0.3984,  # cm
            #     vertical_aperture=0.2952,    # cm
            #     horizontal_aperture_offset=0.0,
            #     vertical_aperture_offset=0.0,
            #     clipping_range=(0.1, 100000.0),  # m
            #     lock_camera=True,
            # spawn = sim_utils.PinholeCameraCfg(
            #     focal_length=0.05,  # cm
            #     focus_distance=400.0,  # m
            #     horizontal_aperture=0.5,  # cm
            #     vertical_aperture=0.4,    # cm
            #     horizontal_aperture_offset=0.0,
            #     vertical_aperture_offset=0.0,
            #     clipping_range=(0.1, 100000.0),  # m
            #     lock_camera=True,
            # ),
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=0.03,  # cm
                focus_distance=400.0,  # m
                horizontal_aperture=1.0,  # cm
                vertical_aperture=0.8,    # cm
                horizontal_aperture_offset=0.0,
                vertical_aperture_offset=0.0,
                clipping_range=(0.1, 100000.0),  # m
                lock_camera=True,
            ),

            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )