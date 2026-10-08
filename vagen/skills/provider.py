"""Skill retrieval, rendering, reward, and distillation orchestration."""

import logging
import os
from typing import Any, Dict, List, Optional

from .base import SkillProvider
from .renderer import (
    compose_skill_image,
    normalize_render_config,
    normalize_render_scale,
    render_frames_with_path,
)

logger = logging.getLogger(__name__)

TASK_DESCRIPTIONS = {
    "sokoban": (
        "Push all boxes onto target squares. You can only push (not pull). "
        "Avoid pushing boxes into corners or against walls."
    ),
    "frozenlake": (
        "Navigate the player to the goal tile while avoiding holes. "
        "Move only on frozen tiles."
    ),
    "maniskill": (
        "Manipulate objects on the tabletop with pick, place, and push "
        "primitives to reach the goal configuration."
    ),
}

# Per-env_id overrides of the task description for multi-task env types
# (handwritten summaries of each env's instruction, same style as above).
TASK_DESCRIPTION_OVERRIDES = {
    "AlignTwoCube": (
        "Move both cubes to x=0 while keeping their y positions, so that "
        "they align along the y-axis."
    ),
    "PlaceTwoCube": (
        "Place the red cube on the left target and the green cube on the "
        "right target."
    ),
    "StackThreeCube": (
        "Stack the red cube on top of the green cube, then stack the purple "
        "cube on top of the red cube."
    ),
    "PutAppleInDrawer": (
        "Put the apple inside the open drawer, then push the drawer handle "
        "until the drawer is closed."
    ),
    "SwapTwoCube": (
        "Swap the cubes: place the red cube on the right target and the "
        "green cube on the left target, using the middle buffer zone to "
        "temporarily hold a cube."
    ),
}

# env_name -> canonical env_type for skill registries
ENV_TYPE_ALIASES = {"remoteenv": "maniskill"}

SKILL_INTRO = (
    "\n\n[Skill Reference]:\n"
    "The following shows a successful solution for a similar task. "
    "Use it as guidance for your approach.\n"
)

# Suffix appended to the strategy text when a skill is created: it tells the
# policy consuming the skill how to use it.
# Keep the maniskill entry in sync with oracle.GROUNDING_HINT.
STRATEGY_SUFFIXES = {
    "maniskill": (
        "Follow this action sequence, but adapt all coordinates to the "
        "current scene."
    ),
}


def append_strategy_suffix(env_type: str, strategy: str) -> str:
    suffix = STRATEGY_SUFFIXES.get(env_type)
    if not strategy or not suffix or suffix in strategy:
        return strategy
    return strategy.rstrip() + " " + suffix


class ViSkillProvider(SkillProvider):

    def __init__(self, config: dict):
        self.config = config

        # Nested config sections
        lib_cfg = config.get("library", {})
        ret_cfg = config.get("retrieval", {})
        util_cfg = config.get("utility", {})
        reward_cfg = config.get("reward", {})
        distill_cfg = config.get("distill", {})
        render_cfg = config.get("render", {})

        # Library
        self.library_path = lib_cfg.get("path", "./skill_library")
        self._lib_max_size = lib_cfg.get("max_size", 32)

        # Retrieval
        self._similarity_weight = ret_cfg.get("similarity_weight", 0.9)
        self.min_score = ret_cfg.get("min_score", 0.8)

        # Utility
        self._ema_alpha = util_cfg.get("ema_alpha", 0.1)
        self._utility_initial = util_cfg.get("initial", 0.5)

        # Reward
        self.reward_mode = reward_cfg.get("mode", "success")
        self.format_reward = reward_cfg.get("format_reward", 0.1)
        self.guided_weight = reward_cfg.get("guided_weight", 0.3)
        self._step_reward = reward_cfg.get("step_reward", 0.05)
        self._unsolvable_penalty = reward_cfg.get("unsolvable_penalty", -0.2)
        self.process_reward_enable = self.reward_mode == "bfs_process"

        # Distill
        self.distill_mode = distill_cfg.get("mode", "self")
        self.distill_quality_ratio = distill_cfg.get("quality_ratio", 1.5)
        self.distill_min_score = distill_cfg.get("min_score", 0.9)

        # Render
        strat_cfg, traj_cfg = normalize_render_config(render_cfg)
        self.render_strategy_enable = strat_cfg.get("enable", True)
        self.render_strategy_type = strat_cfg.get("type", "vision")  # vision / text

        self.render_trajectory_enable = traj_cfg.get("enable", True)
        self.render_action_type = traj_cfg.get("action_type", "text")
        self.render_show_title = traj_cfg.get("show_title", True)
        self.render_draw_path = traj_cfg.get("draw_path", False)
        self.render_thumbnail_size = traj_cfg.get("thumbnail_size")
        self.render_scale = normalize_render_scale(render_cfg.get("scale", 1.0))

        # Lazy-initialized components
        self.library = None
        self._process_reward_computer = None
        self._renderer = None
        self._env_type_initialized = False
        self.task_description = config.get("task_description", "")

        self._api_distiller = None
        if self.distill_mode == "api":
            from .api_distiller import ViSkillDistiller
            self._api_distiller = ViSkillDistiller(
                model=distill_cfg.get("model", "o3"),
                api_base=distill_cfg.get("api_base"),
                api_key=distill_cfg.get("api_key"),
                max_retries=distill_cfg.get("max_retries", 3),
            )

    def init_env_type(self, env_name: str):
        if self._env_type_initialized:
            return
        self._env_type_initialized = True

        from .features import get_feature_extractor
        from .library import ViSkillLibrary
        from .process_reward import get_process_reward_computer
        from .renderer import get_skill_renderer

        env_type = ENV_TYPE_ALIASES.get(env_name.lower(), env_name.lower())
        self.env_type = env_type
        self.library = ViSkillLibrary(
            feature_extractor=get_feature_extractor(env_type),
            max_size=self._lib_max_size,
            ema_alpha=self._ema_alpha,
            utility_initial=self._utility_initial,
            similarity_weight=self._similarity_weight,
        )

        self._process_reward_computer = get_process_reward_computer(env_type)

        if not self.task_description:
            self.task_description = TASK_DESCRIPTIONS.get(env_type, "")

        try:
            self._renderer = get_skill_renderer(env_type)
        except ValueError:
            logger.warning(f"No renderer for '{env_type}', skill images disabled")

        if self.library_path and os.path.exists(self.library_path):
            self.library.load(self.library_path)

    # ---- Retrieval ----

    def retrieve(self, env_state: dict) -> Optional[dict]:
        if self.library is None or not env_state:
            return None
        if self.library_path:
            self.library.load(self.library_path)
        return self.library.retrieve(env_state)

    def format_for_prompt(self, skill: dict) -> Dict[str, Any]:
        strategy = skill.get("strategy", "") if self.render_strategy_enable else ""
        img = self.library.load_image(skill, self.library_path)

        if img is not None and self.render_strategy_type == "text" and strategy:
            # Trajectory image + strategy as separate text
            return {
                "obs_str": SKILL_INTRO + strategy + "\n<image>",
                "multi_modal_input": {"<image>": [img]},
            }
        if img is not None:
            # Image contains all info (strategy rendered in image)
            return {
                "obs_str": SKILL_INTRO + "<image>",
                "multi_modal_input": {"<image>": [img]},
            }
        if strategy:
            # No image, text-only
            return {"obs_str": SKILL_INTRO + strategy}
        return {}

    # ---- Reward ----

    def compute_process_reward(self, state_before: dict, state_after: dict) -> float:
        if self._process_reward_computer is None or not state_before or not state_after:
            return 0.0
        return self._process_reward_computer.compute(
            state_before, state_after,
            step_reward=self._step_reward,
            unsolvable_penalty=self._unsolvable_penalty,
        )

    def bfs_distance(self, env_state: dict) -> Optional[int]:
        if self._process_reward_computer is None or not env_state:
            return None
        return self._process_reward_computer.distance_to_goal(env_state)

    # ---- Distillation ----

    def should_distill(
        self,
        total_actions: int,
        bfs_steps: Optional[int],
        skill_score: Optional[float],
    ) -> bool:
        if bfs_steps is None or bfs_steps == 0:
            return False
        if total_actions / bfs_steps > self.distill_quality_ratio:
            return False
        if skill_score is not None and skill_score >= self.distill_min_score:
            return False
        return True

    def render_trajectory(
        self, frames, actions, positions=None, strategy="", artifact=None
    ) -> Optional[Any]:
        strategy_in_image = (
            self.render_strategy_enable and self.render_strategy_type == "vision"
        )
        if not frames:
            return None
        if not self.render_trajectory_enable and not strategy_in_image:
            return None
        rendered = frames
        if self.render_draw_path and positions and self._renderer:
            rendered = render_frames_with_path(frames, positions, self._renderer)
        return compose_skill_image(
            rendered, actions,
            strategy_text=strategy if strategy_in_image else "",
            thumbnail_size=self.render_thumbnail_size,
            show_title=self.render_show_title,
            trajectory_enable=self.render_trajectory_enable,
            strategy_enable=strategy_in_image,
            action_type=self.render_action_type,
            scale=self.render_scale,
        )

    def render_initial_state(self, frame=None, artifact=None) -> Optional[Any]:
        """Render the initial state paired with a retrieved skill."""
        return frame.copy() if hasattr(frame, "copy") else frame

    def task_description_for(self, env_state: dict) -> str:
        """Per-env_id task description when available, else the env_type one."""
        env_id = (env_state or {}).get("env_id")
        return TASK_DESCRIPTION_OVERRIDES.get(env_id, self.task_description)

    def distill_api(self, case_data: dict) -> str:
        if self._api_distiller is None:
            return ""
        result = self._api_distiller.distill(case_data)
        return result.get("strategy", "")

    @property
    def api_distill_enabled(self) -> bool:
        return self._api_distiller is not None

    def extract_features(self, env_state: dict) -> List[int]:
        if self.library is None or not env_state:
            return []
        return self.library.feature_extractor.extract(env_state)

    # ---- Utility ----

    def update_utility(self, skill_id: str, reward: float) -> None:
        if self.library:
            self.library.update_utility(skill_id, reward)

    def save(self) -> None:
        if self.library and self.library_path:
            self.library.save(self.library_path)
