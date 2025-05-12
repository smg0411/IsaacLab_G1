# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    --enable_cameras

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg, CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets import G1_WITH_HAND_CFG

from isaaclab.sensors.camera.utils import save_images_to_file
import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend


from PIL import Image
import numpy as np 
import os.path as osp


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=[0,0,-1.05]))

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = G1_WITH_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",
    #     # update_period=0.1,
    #     update_period=0.1,
    #     height=1080,
    #     width=1920,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     # ),
    #     spawn = sim_utils.PinholeCameraCfg(
    #         focal_length=0.193,  # cm
    #         focus_distance=400.0,  # m
    #         horizontal_aperture=0.3984,  # cm
    #         vertical_aperture=0.2952,    # cm
    #         horizontal_aperture_offset=0.0,
    #         vertical_aperture_offset=0.0,
    #         clipping_range=(0.1, 100000.0),  # m
    #         lock_camera=True,
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/camera",
        # update_period=0.1,
        update_period=0.1,
        height=1080,
        width=1920,
        data_types=["rgb", "distance_to_image_plane"],
        # spawn=sim_utils.PinholeCameraCfg(
        #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        # ),
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=0.193,  # cm
            focus_distance=400.0,  # m
            horizontal_aperture=0.3984,  # cm
            vertical_aperture=0.2952,    # cm
            horizontal_aperture_offset=0.0,
            vertical_aperture_offset=0.0,
            clipping_range=(0.1, 100000.0),  # m
            lock_camera=True,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    output_dir = osp.join(osp.dirname(osp.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=scene["camera"].cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=scene["camera"].cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=scene["camera"].cfg.colorize_semantic_segmentation,
    )

    camera_idx = 1

    # Simulate physics
    while simulation_app.is_running():
        ##
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        print("-------------------------------")
        print(scene["camera"])
        print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
        print("Received shape of depth image: ", scene["camera"].data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        single_cam_data = convert_dict_to_backend(
                {k: v[camera_idx] for k, v in scene["camera"].data.output.items()}, backend="numpy"
            )
        rep_output = {"annotators":{}}
        for key, data in zip(single_cam_data.keys(), single_cam_data.values()):
            rep_output["annotators"][key] = {"render_product":{"data":data}}

        rep_output["trigger_outputs"] = {"on_time": scene["camera"].frame[camera_idx]}
        rep_writer.write(rep_output)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
