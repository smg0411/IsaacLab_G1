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

@configclass
class JointPosEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # G1 + FTP Hands 로봇 설정
        self.scene.robot = G1_WITH_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # G1용 arm 제어 (관절 제어)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # G1 손가락 이진 제어 (open/close)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*_index_.*", ".*_thumb_.*", ".*_middle_.*", ".*_ring_.*", ".*_little_.*"],
            open_command_expr={".*": 0.0},
            close_command_expr={".*": 0.8},
        )

        # EEF 기준 링크 지정 (왼손 손바닥 기준)
        self.commands.object_pose.body_name = "left_wrist_pitch_link"

        # 들어 올릴 큐브 설정
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # EE Frame Transformer 설정 (선택적 시각화)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_wrist_pitch_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_index_1",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.05]),
                ),
            ],
        )