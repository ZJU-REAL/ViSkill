import asyncio
import logging
import os
import traceback
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from PIL import Image
from verl.utils.rollout_trace import rollout_trace_op

from .gym_agent_loop import (
    GymAgentLoop as _BaseGymAgentLoop,
    AgentData as _BaseAgentData,
    AgentState,
    convert_obs_to_content,
    extract_success,
    _normalize_images,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentData(_BaseAgentData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieved_skill: Optional[dict] = None
        self.skill_score: Optional[float] = None
        self.cached_env_state: Optional[dict] = None
        self.initial_env_state: Optional[dict] = None
        self.trajectory_frames: List[Image.Image] = []
        self.trajectory_actions: List[str] = []
        self.trajectory_positions: List[Any] = []
        self.skill_artifact: Optional[dict] = None
        self.validation_skill_image: Optional[Image.Image] = None
        self.validation_initial_state_image: Optional[Image.Image] = None


class GymAgentLoop(_BaseGymAgentLoop):

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if getattr(cls, "_skill_initialized", False):
            return
        super().init_class(config, tokenizer, processor, **kwargs)
        cls._skill_initialized = True

        cls.skill_provider = None
        skill_cfg = config.get("skill_system")
        if skill_cfg and skill_cfg.get("enable", True):
            from vagen.skills import get_skill_provider
            cls.skill_provider = get_skill_provider(dict(skill_cfg))

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs):
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        env_name = kwargs["env_name"]
        if env_name not in self.env_registry:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths.keys())}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_path)
            self.env_registry[env_name] = getattr(module, class_name)

        env_cls = self.env_registry[env_name]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        is_validation = bool(kwargs.get("validate", False))
        self.env_max_turns = kwargs.get("max_turns")
        env = env_cls(env_config=env_config)

        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        messages: List[Dict[str, Any]] = []
        image_data: List[Image.Image] = []

        # --- System prompt (VAGEN style: direct injection) ---
        if sys_obs:
            messages.append({"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)})
            sys_imgs = sys_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(sys_imgs))

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=env_name,
        )

        # --- Skill retrieval + injection ---
        if self.skill_provider is not None and hasattr(env, "get_state_for_skill"):
            self.skill_provider.init_env_type(env_name)
            env_state = env.get_state_for_skill()
            if asyncio.iscoroutine(env_state):
                env_state = await env_state
            agent_data.cached_env_state = env_state
            agent_data.initial_env_state = env_state

            if not env_state:
                logger.warning(
                    "Skill state unavailable for env=%s seed=%s; "
                    "continuing without retrieval",
                    env_name,
                    seed,
                )

            if self.skill_provider.env_type != "maniskill":
                init_traj_imgs = init_obs.get("multi_modal_input", {}).get("<image>", []) or []
                agent_data.trajectory_frames.extend(_normalize_images(init_traj_imgs))
                pos = self._extract_position(env_state)
                if pos:
                    agent_data.trajectory_positions.append(pos)

            skill = self.skill_provider.retrieve(env_state)
            if (
                skill
                and skill.get("score", 0) >= self.skill_provider.min_score
                and self._inject_skill(agent_data, skill, **kwargs)
            ):
                agent_data.retrieved_skill = skill
                agent_data.skill_score = skill["score"]

        # --- Initial observation ---
        # Append this after skill injection so image_data follows the exact
        # placeholder order: system images, skill images, observation images.
        if init_obs:
            agent_data.messages.append({
                "role": "user",
                "content": convert_obs_to_content(init_obs, **kwargs),
            })
            init_imgs = init_obs.get("multi_modal_input", {}).get("<image>", []) or []
            init_imgs = _normalize_images(init_imgs)
            agent_data.image_data.extend(init_imgs)
            if (
                is_validation
                and agent_data.retrieved_skill is not None
                and init_imgs
            ):
                agent_data.validation_initial_state_image = init_imgs[0].copy()

        # --- State machine ---
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)
            else:
                state = AgentState.TERMINATED

        if (
            (
                agent_data.traj_success
                or (
                    is_validation
                    and agent_data.retrieved_skill is not None
                )
            )
            and self.skill_provider is not None
            and self.skill_provider.env_type == "maniskill"
            and hasattr(env, "get_skill_artifact")
        ):
            try:
                artifact = env.get_skill_artifact()
                if asyncio.iscoroutine(artifact):
                    artifact = await artifact
                if artifact:
                    agent_data.skill_artifact = artifact
                    agent_data.trajectory_frames = list(artifact.get("frames", []))
            except Exception as exc:
                logger.warning("Failed to collect skill artifact: %s", exc)

        await env.close()

        # --- Self-distill (independent window, only for quality successes) ---
        distilled_strategy = ""
        if self._should_self_distill(agent_data):
            distilled_strategy = await self._self_distill(agent_data, sampling_params, env_name)

        # --- Compute final reward ---
        reward_score = self._compute_reward(agent_data)

        # --- Finalize output ---
        resp_len = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.prompt_ids[:len(agent_data.prompt_ids) - resp_len]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data else {}

        from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
        return AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[:self.response_length],
            response_mask=agent_data.response_mask[:self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[:self.response_length]
                if agent_data.response_logprobs else None
            ),
            reward_score=reward_score,
            num_turns=agent_data.env_turns,
            metrics=agent_data.metrics,
            extra_fields={
                "reward_extra_info": {
                    "traj_success": float(agent_data.traj_success),
                },
                "traj_success": agent_data.traj_success,
                "retrieved_skill_id": agent_data.retrieved_skill["id"] if agent_data.retrieved_skill else None,
                "skill_score": agent_data.skill_score,
                "seed": seed,
                "env_name": env_name,
                "initial_env_state": agent_data.initial_env_state,
                "trajectory_actions": agent_data.trajectory_actions,
                "trajectory_frames": agent_data.trajectory_frames,
                "trajectory_positions": agent_data.trajectory_positions,
                "skill_artifact": agent_data.skill_artifact,
                "distilled_strategy": distilled_strategy,
                "validation_skill_image": agent_data.validation_skill_image,
                "validation_initial_state_image": (
                    agent_data.validation_initial_state_image
                ),
            },
        )

    # ---- Skill injection (VAGEN style: append to system prompt) ----

    def _inject_skill(self, agent_data: AgentData, skill: dict, **kwargs) -> bool:
        skill_obs = self.skill_provider.format_for_prompt(skill)
        if not skill_obs or "obs_str" not in skill_obs:
            return False
        skill_content = convert_obs_to_content(skill_obs, **kwargs)
        if not agent_data.messages or agent_data.messages[0]["role"] != "system":
            return False

        sys_content = agent_data.messages[0]["content"]
        if isinstance(sys_content, list):
            sys_content.extend(skill_content)
        elif isinstance(sys_content, str):
            agent_data.messages[0]["content"] = [
                {"type": "text", "text": sys_content}
            ] + skill_content
        else:
            return False

        skill_imgs = skill_obs.get("multi_modal_input", {}).get("<image>", []) or []
        skill_imgs = _normalize_images(skill_imgs)
        agent_data.image_data.extend(skill_imgs)
        if kwargs.get("validate", False) and skill_imgs:
            agent_data.validation_skill_image = skill_imgs[0].copy()
        return True

    # ---- Reward computation ----

    def _compute_reward(self, agent_data: AgentData) -> float:
        from vagen.skills.reward import compute_reward
        mode = self.skill_provider.reward_mode if self.skill_provider else "success"
        skill_traj = (
            agent_data.retrieved_skill.get("trajectory")
            if agent_data.retrieved_skill else None
        )
        model_traj = agent_data.trajectory_actions
        if self.skill_provider and self.skill_provider.env_type == "maniskill":
            # Match the action sequence only; coordinates are adapted per scene.
            if skill_traj:
                skill_traj = [a.split("(")[0] for a in skill_traj]
            model_traj = [a.split("(")[0] for a in model_traj]
        guided_weight = self.skill_provider.guided_weight if self.skill_provider else 0.5
        return compute_reward(
            env_rewards=agent_data.env_rewards,
            mode=mode,
            traj_success=agent_data.traj_success,
            skill_trajectory=skill_traj,
            model_trajectory=model_traj,
            guided_weight=guided_weight,
        )

    # ---- Env interaction ----

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
        except Exception as exc:
            logger.error("Env step failed in '%s': %s", agent_data.env_name, exc)
            logger.error(traceback.format_exc())
            obs, reward, done, info = (
                {"obs_str": "Environment Error"}, 0.0, True, {"traj_success": False}
            )

        process_r = await self._step_process_reward(agent_data)
        agent_data.env_rewards.append(float(reward) + process_r)
        agent_data.traj_success = extract_success(info)
        agent_data.env_turns += 1
        self._collect_trajectory(agent_data, obs, info)

        if done or agent_data.traj_success:
            return AgentState.TERMINATED
        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            return AgentState.TERMINATED
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        new_images = obs.get("multi_modal_input", {}).get("<image>", []) or []
        new_images = _normalize_images(new_images)
        await self._append_user_turn(agent_data, obs, new_images or None, **kwargs)
        return AgentState.GENERATING

    async def _step_process_reward(self, agent_data: AgentData) -> float:
        if (
            self.skill_provider is None
            or not self.skill_provider.process_reward_enable
            or agent_data.cached_env_state is None
            or not hasattr(agent_data.env, "get_state_for_skill")
        ):
            return 0.0
        new_state = agent_data.env.get_state_for_skill()
        if asyncio.iscoroutine(new_state):
            new_state = await new_state
        r = self.skill_provider.compute_process_reward(agent_data.cached_env_state, new_state)
        agent_data.cached_env_state = new_state
        return r

    def _collect_trajectory(self, agent_data: AgentData, obs: dict, info: dict) -> None:
        if self.skill_provider is None:
            return
        actions = info.get("actions", [])
        if self.skill_provider.env_type != "maniskill":
            step_frames = info.get("step_frames", [])
            if step_frames:
                agent_data.trajectory_frames.extend(step_frames)
            elif actions:
                traj_imgs = obs.get("multi_modal_input", {}).get("<image>", []) or []
                agent_data.trajectory_frames.extend(_normalize_images(traj_imgs))
        if actions:
            agent_data.trajectory_actions.extend(actions)
        step_positions = info.get("step_positions", [])
        if step_positions:
            agent_data.trajectory_positions.extend(step_positions)
        elif agent_data.cached_env_state is not None:
            pos = self._extract_position(agent_data.cached_env_state)
            if pos:
                agent_data.trajectory_positions.append(pos)

    # ---- Tokenize helpers ----

    async def _tokenize_messages(
        self, messages: List[Dict], images: Optional[List[Image.Image]] = None
    ) -> List[int]:
        if self.processor is not None:
            raw_text = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(
                text=[raw_text], images=images or None, return_tensors="pt"
            )
            return model_inputs.pop("input_ids").squeeze(0).tolist()

        if images:
            raise ValueError("Images provided but `processor` is None.")
        from .gym_agent_loop import _flatten_text_only_content
        flat_messages = [_flatten_text_only_content(m) for m in messages]
        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                flat_messages, add_generation_prompt=True,
                tokenize=True, return_dict=False,
                **self.apply_chat_template_kwargs,
            ),
        )

    async def _append_user_turn(
        self, agent_data: AgentData, obs: Dict[str, Any],
        images: Optional[List[Image.Image]] = None, **kwargs,
    ) -> None:
        content = convert_obs_to_content(obs, **kwargs)
        msg = {"role": "user", "content": content}
        agent_data.messages.append(msg)

        _placeholder = {"role": "system", "content": "placeholder"}
        suffix_ids = await self._tokenize_messages([_placeholder, msg], images)
        suffix_ids = suffix_ids[len(self.system_prompt_prefix):]

        agent_data.prompt_ids += suffix_ids
        agent_data.response_mask += [0] * len(suffix_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(suffix_ids)
        if images:
            agent_data.image_data.extend(images)

    # ---- Self-distill (independent generation window) ----

    async def _self_distill(
        self, agent_data: AgentData, sampling_params: Dict[str, Any], env_name: str
    ) -> str:
        from vagen.skills.distill_prompt import build_distill_prompt, build_distill_images
        from vagen.skills.api_distiller import parse_strategy

        actions = agent_data.trajectory_actions
        init_frame = agent_data.trajectory_frames[0] if agent_data.trajectory_frames else None
        if self.skill_provider.env_type == "maniskill" and agent_data.skill_artifact:
            actions = agent_data.skill_artifact.get("actions", [])
        trajectory_image = self.skill_provider.render_trajectory(
            agent_data.trajectory_frames, actions,
            positions=agent_data.trajectory_positions,
            artifact=agent_data.skill_artifact,
        )

        prompt_text = build_distill_prompt(
            self.skill_provider.task_description_for(agent_data.initial_env_state),
            actions,
            env_type=self.skill_provider.env_type,
            has_trajectory_image=(trajectory_image is not None),
            env_id=(agent_data.initial_env_state or {}).get("env_id", ""),
        )
        images = build_distill_images(init_frame, trajectory_image)

        obs = {"obs_str": prompt_text}
        if images:
            obs["multi_modal_input"] = {"<image>": images}

        msg = {"role": "user", "content": convert_obs_to_content(obs, env_name=env_name, config={})}
        prompt_ids = await self._tokenize_messages([msg], images)

        prefix_ids = self.tokenizer.encode("<strategy>", add_special_tokens=False)
        prompt_ids = prompt_ids + prefix_ids

        params = sampling_params.copy()
        params["max_new_tokens"] = min(256, params.get("max_new_tokens", 256))
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=params,
            image_data=images or None,
        )

        raw_output = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        return parse_strategy("<strategy>" + raw_output)

    # ---- Helpers ----

    def _should_self_distill(self, agent_data: AgentData) -> bool:
        if self.skill_provider is None or self.skill_provider.distill_mode != "self":
            return False
        if not agent_data.traj_success:
            return False
        env_type = self.skill_provider.env_type
        actions = agent_data.trajectory_actions
        if env_type == "maniskill":
            if not agent_data.skill_artifact:
                return False
            actions = agent_data.skill_artifact.get("actions") or []
        if not actions:
            return False
        bfs = self.skill_provider.bfs_distance(agent_data.initial_env_state)
        return self.skill_provider.should_distill(
            len(actions), bfs, agent_data.skill_score
        )

    @staticmethod
    def _extract_position(env_state: dict) -> Optional[tuple]:
        if "player_pos" in env_state:
            return tuple(env_state["player_pos"])
        if "room_state" in env_state:
            pos = np.argwhere(env_state["room_state"] == 5)
            if len(pos) > 0:
                return tuple(pos[0])
        return None
