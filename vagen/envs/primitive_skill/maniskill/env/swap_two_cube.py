from typing import Any, Dict, Union

import numpy as np
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat


@register_env("SwapTwoCube", max_episode_steps=3e3)
class SwapTwoCubeEnv(BaseEnv):
    """Swap two cubes between fixed left/right targets via a buffer zone."""

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Fetch]
    skill_config = None
    vlm_info_keys = []
    state_keys = [
        "red_cube_position",
        "green_cube_position",
        "red_target_position",
        "green_target_position",
        "buffer_position",
    ]

    def __init__(
        self,
        stage=0,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.stage = stage
        self.workspace_x = [-0.10, 0.15]
        self.workspace_y = [-0.2, 0.2]
        self.workspace_z = [0.01, 0.2]
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.goal_radius = 0.10
        self.buffer_pos = torch.tensor([0.025, 0.0, 0.0])
        self.left_target_pos = torch.tensor([0.08, -0.12, 0.0])
        self.right_target_pos = torch.tensor([0.08, 0.12, 0.0])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def task_skill_indices(self):
        return {
            0: "pick",
            1: "place",
            2: "push",
        }

    def instruction(self):
        return (
            "Please swap the cubes: place the red cube onto the right target "
            "and the green cube onto the left target. "
            "Hint: you can use the buffer zone (the marker in the middle) to "
            "temporarily hold a cube."
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[1, 0.0, 0.6], target=[-0.2, 0.0, 0.2]
        )
        return [
            CameraConfig("base_camera", pose, 300, 300, np.pi / 2, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1, 0.0, 0.6], [-0.2, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 300, 300, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.red_cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="red_cube",
        )
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="green_cube",
        )

        self.red_target = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="red_target",
            add_collision=False,
            body_type="kinematic",
            initial_pose=Pose.create_from_pq(p=self.right_target_pos.clone()),
        )
        self.green_target = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="green_target",
            add_collision=False,
            body_type="kinematic",
            initial_pose=Pose.create_from_pq(p=self.left_target_pos.clone()),
        )
        self.buffer_zone = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="buffer_zone",
            add_collision=False,
            body_type="kinematic",
            initial_pose=Pose.create_from_pq(p=self.buffer_pos.clone()),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)

            left_region = np.array(
                [
                    [self.workspace_x[0], self.workspace_y[0]],
                    [self.workspace_x[1], -0.05],
                ]
            )
            right_region = np.array(
                [
                    [self.workspace_x[0], 0.05],
                    [self.workspace_x[1], self.workspace_y[1]],
                ]
            )
            sampler_left = randomization.UniformPlacementSampler(
                bounds=left_region, batch_size=batch_size
            )
            sampler_right = randomization.UniformPlacementSampler(
                bounds=right_region, batch_size=batch_size
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.02

            red_cube_xy = sampler_left.sample(radius, 100)
            green_cube_xy = sampler_right.sample(radius, 100, verbose=False)

            xyz = torch.zeros((batch_size, 3))
            xyz[:, 2] = 0.02

            xyz[:, :2] = red_cube_xy
            orientation = randomization.random_quaternions(
                batch_size, lock_x=True, lock_y=True, lock_z=False
            )
            self.red_cube.set_pose(
                Pose.create_from_pq(p=xyz.clone(), q=orientation)
            )

            xyz[:, :2] = green_cube_xy
            orientation = randomization.random_quaternions(
                batch_size, lock_x=True, lock_y=True, lock_z=False
            )
            self.green_cube.set_pose(
                Pose.create_from_pq(p=xyz.clone(), q=orientation)
            )

            target_orientation = euler2quat(0, np.pi / 2, 0)
            self.red_target.set_pose(
                Pose.create_from_pq(
                    p=self.right_target_pos.clone(), q=target_orientation
                )
            )
            self.green_target.set_pose(
                Pose.create_from_pq(
                    p=self.left_target_pos.clone(), q=target_orientation
                )
            )
            self.buffer_zone.set_pose(
                Pose.create_from_pq(
                    p=self.buffer_pos.clone(), q=target_orientation
                )
            )

    def _is_at_target(self, cube_pos, target_pos):
        distance = torch.norm(
            cube_pos[..., :2] - target_pos[..., :2], dim=-1
        )
        return distance <= self.goal_radius

    def evaluate(self):
        red_position = self.red_cube.pose.p
        green_position = self.green_cube.pose.p

        if self.agent is None:
            not_reached = torch.zeros(
                red_position.shape[:-1], dtype=torch.bool, device=self.device
            )
            return {
                "red_cube_position": red_position,
                "green_cube_position": green_position,
                "red_target_position": self.red_target.pose.p,
                "green_target_position": self.green_target.pose.p,
                "buffer_position": self.buffer_zone.pose.p,
                "is_red_cube_grasped": not_reached,
                "is_green_cube_grasped": not_reached,
                "is_red_at_target": not_reached,
                "is_green_at_target": not_reached,
                "stage_0_success": not_reached,
                "stage_1_success": not_reached,
                "stage_2_success": not_reached,
                "stage_3_success": not_reached,
                "success": not_reached,
            }

        red_grasped = self.agent.is_grasping(self.red_cube)
        green_grasped = self.agent.is_grasping(self.green_cube)
        red_at_target = self._is_at_target(
            red_position, self.red_target.pose.p
        )
        green_at_target = self._is_at_target(
            green_position, self.green_target.pose.p
        )

        any_grasped = red_grasped | green_grasped
        one_at_target = (
            (red_at_target & ~red_grasped)
            | (green_at_target & ~green_grasped)
        )
        one_done_other_grasped = (
            ((red_at_target & ~red_grasped) & green_grasped)
            | ((green_at_target & ~green_grasped) & red_grasped)
        )
        both_at_target = (
            red_at_target
            & green_at_target
            & ~red_grasped
            & ~green_grasped
        )

        return {
            "red_cube_position": red_position,
            "green_cube_position": green_position,
            "red_target_position": self.red_target.pose.p,
            "green_target_position": self.green_target.pose.p,
            "buffer_position": self.buffer_zone.pose.p,
            "is_red_cube_grasped": red_grasped,
            "is_green_cube_grasped": green_grasped,
            "is_red_at_target": red_at_target,
            "is_green_at_target": green_at_target,
            "stage_0_success": any_grasped,
            "stage_1_success": one_at_target,
            "stage_2_success": one_done_other_grasped,
            "stage_3_success": both_at_target,
            "success": both_at_target.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        assert "state" in self.obs_mode
        return {
            "red_cube_position": info["red_cube_position"],
            "green_cube_position": info["green_cube_position"],
            "red_target_position": info["red_target_position"],
            "green_target_position": info["green_target_position"],
            "buffer_position": info["buffer_position"],
            "is_red_cube_grasped": info["is_red_cube_grasped"],
            "is_green_cube_grasped": info["is_green_cube_grasped"],
        }

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return torch.zeros_like(
            info["success"], dtype=torch.float32, device=self.device
        )

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 80.0

    def task_fail(self, info: Dict):
        for cube in ("red_cube", "green_cube"):
            position = info[f"{cube}_position"]
            outside_x = (position[..., 0] < self.workspace_x[0]) | (
                position[..., 0] > self.workspace_x[1]
            )
            outside_y = (position[..., 1] < self.workspace_y[0]) | (
                position[..., 1] > self.workspace_y[1]
            )
            below_table = position[..., 2] < 0
            if bool(torch.any(outside_x | outside_y | below_table)):
                return True
        return False

    def skill_reward(self, prev_info, cur_info, action, **kwargs):
        return 0.0

