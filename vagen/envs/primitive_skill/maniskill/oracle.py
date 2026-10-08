"""Scripted oracle planners for the ManiSkill primitive-skill tasks.

Shared by test/extract_success (skill extraction), vagen.skills.cold_start
(cold-start library generation) and the skill feature extractor / solver.
Pure Python + numpy: safe to import without ManiSkill installed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

ALL_ENV_IDS = (
    "AlignTwoCube",
    "PlaceTwoCube",
    "StackThreeCube",
    "PutAppleInDrawer",
    "SwapTwoCube",
)

# Hard partition index appended to feature vectors so that skills never
# transfer across tasks (Align/Place/Swap share the same object count).
TASK_INDEX = {env_id: index for index, env_id in enumerate(ALL_ENV_IDS)}

GROUNDING_HINT = (
    "Follow this action sequence, but adapt all coordinates to the current scene."
)


def position_mm(info: dict, key: str) -> Tuple[int, int, int]:
    value = np.asarray(info[key]).reshape(-1)
    if value.size != 3:
        raise ValueError(f"Expected a 3D position for {key}, got shape {value.shape}")
    return tuple(int(round(float(v) * 1000)) for v in value)


def _action(name: str, *coordinates: int) -> str:
    return f"{name}({','.join(str(int(v)) for v in coordinates)})"


def _pick(position: Tuple[int, int, int]) -> str:
    return _action("pick", *position)


def _place(position: Tuple[int, int, int]) -> str:
    return _action("place", *position)


def _ground_place(position: Tuple[int, int, int]) -> str:
    return _place((position[0], position[1], 25))


def _align_plan(info: dict) -> dict:
    red = position_mm(info, "red_cube_position")
    green = position_mm(info, "green_cube_position")
    actions = [
        _pick(red),
        _ground_place((0, red[1], 0)),
        _pick(green),
        _ground_place((0, green[1], 0)),
    ]
    return {
        "actions": actions,
        "semantic_actions": [
            "pick red cube",
            "place red cube at x=0",
            "pick green cube",
            "place green cube at x=0",
        ],
        "strategy": (
            "Move both cubes to x=0 while keeping their y positions. "
            + GROUNDING_HINT
        ),
    }


def _place_plan(info: dict) -> dict:
    red = position_mm(info, "red_cube_position")
    green = position_mm(info, "green_cube_position")
    left = position_mm(info, "left_target_position")
    right = position_mm(info, "right_target_position")
    return {
        "actions": [
            _pick(red),
            _ground_place(left),
            _pick(green),
            _ground_place(right),
        ],
        "semantic_actions": [
            "pick red cube",
            "place red cube on left target",
            "pick green cube",
            "place green cube on right target",
        ],
        "strategy": (
            "Place the red cube on the left target, then the green cube "
            "on the right target. "
            + GROUNDING_HINT
        ),
    }


def _stack_plan(info: dict) -> dict:
    red = position_mm(info, "red_cube_position")
    green = position_mm(info, "green_cube_position")
    purple = position_mm(info, "purple_cube_position")
    cube_size = int(round(float(np.asarray(info["cube_size"]).reshape(-1)[0]) * 1000))
    place_offset = cube_size + 5
    return {
        "actions": [
            _pick(red),
            _place((green[0], green[1], green[2] + place_offset)),
            _pick(purple),
            _place((green[0], green[1], green[2] + cube_size + place_offset)),
        ],
        "semantic_actions": [
            "pick red cube",
            "place red cube on green cube",
            "pick purple cube",
            "place purple cube on red cube",
        ],
        "strategy": (
            "Build the stack from bottom to top: green, red, then purple. "
            + GROUNDING_HINT
        ),
    }


def _drawer_plan(info: dict) -> dict:
    apple = position_mm(info, "apple_position")
    drawer = position_mm(info, "drawer_position")
    handle = position_mm(info, "drawer_handle_position")
    open_mm = int(round(float(np.asarray(info["drawer_open_value"]).reshape(-1)[0]) * 1000))
    insert_position = (drawer[0] + 50, drawer[1], drawer[2] + 150)
    closed_handle_y = handle[1] - open_mm + 50
    return {
        "actions": [
            _pick(apple),
            _place(insert_position),
            _action(
                "push",
                *handle,
                handle[0],
                closed_handle_y,
                handle[2],
            ),
        ],
        "semantic_actions": [
            "pick apple",
            "place apple inside drawer",
            "push drawer handle closed",
        ],
        "strategy": (
            "Put the apple inside the open drawer, release it, then push the "
            "handle closed. "
            + GROUNDING_HINT
        ),
    }


def _swap_plan(info: dict) -> dict:
    red = position_mm(info, "red_cube_position")
    green = position_mm(info, "green_cube_position")
    red_target = position_mm(info, "red_target_position")
    green_target = position_mm(info, "green_target_position")
    buffer = position_mm(info, "buffer_position")
    return {
        "actions": [
            _pick(red),
            _ground_place(buffer),
            _pick(green),
            _ground_place(green_target),
            _pick((buffer[0], buffer[1], 20)),
            _ground_place(red_target),
        ],
        "semantic_actions": [
            "pick red cube",
            "place red cube on buffer",
            "pick green cube",
            "place green cube on left target",
            "pick red cube from buffer",
            "place red cube on right target",
        ],
        "strategy": (
            "Use the center buffer to free the left target, then swap the two "
            "cubes. "
            + GROUNDING_HINT
        ),
    }


ORACLES = {
    "AlignTwoCube": _align_plan,
    "PlaceTwoCube": _place_plan,
    "StackThreeCube": _stack_plan,
    "PutAppleInDrawer": _drawer_plan,
    "SwapTwoCube": _swap_plan,
}

# Fixed optimal action count per task (length of the oracle plan), used as
# the reference solution length L* for distillation quality gating.
OPTIMAL_STEPS = {
    "AlignTwoCube": 4,
    "PlaceTwoCube": 4,
    "StackThreeCube": 4,
    "PutAppleInDrawer": 3,
    "SwapTwoCube": 6,
}

STATE_KEYS = {
    "AlignTwoCube": ("red_cube_position", "green_cube_position"),
    "PlaceTwoCube": (
        "red_cube_position",
        "green_cube_position",
        "left_target_position",
        "right_target_position",
    ),
    "StackThreeCube": (
        "red_cube_position",
        "green_cube_position",
        "purple_cube_position",
    ),
    "PutAppleInDrawer": (
        "apple_position",
        "drawer_position",
        "drawer_handle_position",
    ),
    "SwapTwoCube": (
        "red_cube_position",
        "green_cube_position",
        "red_target_position",
        "green_target_position",
        "buffer_position",
    ),
}

FEATURE_KEYS = {
    "AlignTwoCube": ("red_cube_position", "green_cube_position"),
    "PlaceTwoCube": ("red_cube_position", "green_cube_position"),
    "StackThreeCube": (
        "red_cube_position",
        "green_cube_position",
        "purple_cube_position",
    ),
    "PutAppleInDrawer": ("apple_position",),
    "SwapTwoCube": ("red_cube_position", "green_cube_position"),
}


def extract_structural_features(env_id: str, initial_state: dict) -> List[int]:
    """Return raw millimeter xy coordinates for task-relevant movable objects."""
    features = []
    for key in FEATURE_KEYS[env_id]:
        x, y, _ = initial_state[key]
        features.extend([x, y])
    return features


def _image_from_obs(obs: dict) -> Image.Image:
    return obs["multi_modal_input"]["<image>"][0].convert("RGB")


def run_oracle_episode(env, env_id: str, seed: int) -> Optional[Dict]:
    """Reset a PrimitiveSkillEnv and execute the scripted oracle plan.

    Returns {frames, plan, initial_info, initial_positions} on success,
    None when the episode ends early or the plan fails to reach the goal.
    """
    obs, _ = env._sync_reset(seed)
    initial_info = dict(env._last_info)
    initial_positions = {
        key: list(position_mm(initial_info, key)) for key in STATE_KEYS[env_id]
    }
    plan = ORACLES[env_id](initial_info)
    frames = [_image_from_obs(obs)]

    for index, action in enumerate(plan["actions"]):
        response = f"<think>Follow the scripted oracle.</think><answer>{action}</answer>"
        obs, _, done, _ = env._sync_step(response)
        frames.append(_image_from_obs(obs))
        if done and index < len(plan["actions"]) - 1:
            return None

    if not bool(np.asarray(env._last_info.get("is_success", False)).all()):
        return None

    return {
        "frames": frames,
        "plan": plan,
        "initial_info": initial_info,
        "initial_positions": initial_positions,
    }
