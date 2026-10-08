"""
Skill-augmented trainer: extends VAGEN trainer with per-step distillation.
Override fit() to inject distill_step() after each rollout.
"""

import logging
import os
import shutil
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.profiler import marked_timer
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

from vagen.ray_trainer import (
    RayPPOTrainer,
    compute_advantage,
    compute_response_mask,
    compute_value_mask,
    compute_data_metrics,
    compute_timing_metrics,
    compute_throughout_metrics,
    compute_custom_metrics,
    apply_kl_penalty,
    pad_dataproto_to_divisor,
    reduce_metrics,
    agg_loss,
    should_save_ckpt_esi,
    RolloutSkip,
    AdvantageEstimator,
    Role,
)

logger = logging.getLogger(__name__)


class RayPPOTrainerSkill(RayPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distill_callback = None

        skill_cfg = self.config.get("skill_system", None)
        if skill_cfg is not None and skill_cfg.get("enable", False):
            from vagen.skills.distill_callback import DistillCallback

            self._maybe_cold_start(dict(skill_cfg))
            self._distill_callback = DistillCallback(dict(skill_cfg))
            logger.info("Skill distillation callback initialized")

    @staticmethod
    def _maybe_cold_start(skill_cfg: dict):
        cold_cfg = skill_cfg.get("cold_start", {})
        if not cold_cfg.get("enable", False):
            return
        source = cold_cfg.get("source")
        library_path = skill_cfg.get("library", {}).get("path")
        if not source or not library_path:
            return
        if os.path.exists(os.path.join(library_path, "skill_index.json")):
            return
        if not os.path.exists(source):
            logger.warning(f"Cold start source not found: {source}")
            return
        shutil.copytree(source, library_path, dirs_exist_ok=True)
        print(f"[Skill] Cold start: copied from {source} to {library_path}")

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        if "retrieved_skill_id" in batch.non_tensor_batch:
            reward_extra_infos_dict.setdefault(
                "retrieved_skill_id",
                list(batch.non_tensor_batch["retrieved_skill_id"]),
            )
        super()._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger_tracker = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger_tracker.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._flush_image_dumps()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                if not self.concat_multi_turn:
                    num_traj_per_sample = self.config.actor_rollout_ref.rollout.n
                    self._assign_group_and_traj_idx(gen_batch_output, num_traj_per_sample)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # ============ SKILL DISTILLATION (per-step) ============
                    if self._distill_callback is not None:
                        with marked_timer("skill_distill", timing_raw, color="orange"):
                            distill_metrics = self._distill_callback.step(gen_batch_output)
                            metrics.update(distill_metrics)
                    # ======================================================

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if not self.concat_multi_turn:
                            raise NotImplementedError("REMAX advantage estimation is not supported in no-concat mode yet.")
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    if self.concat_multi_turn:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    else:
                        batch = self._post_process_no_concat_batch(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        if not self.concat_multi_turn:
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics
                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
                        batch.batch["value_mask"] = compute_value_mask(batch)

                    with marked_timer("custom_metrics", timing_raw, color="magenta"):
                        custom_train_metrics = compute_custom_metrics(batch, prefix="custom_metrics/train")
                        metrics.update(custom_train_metrics)

                    if self.config.filter.get("enable", False):
                        batch, metrics = FILTER_REGISTRY.get(self.config.filter.name)(batch, metrics, **self.config.filter.filter_kwargs)
                        if self.config.trainer.balance_batch:
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"After filtering: Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                            self._balance_batch(batch, metrics=metrics, logging_prefix="filtered_global_seqlen")

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                should_save_ckpt = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                )
                should_upload_hf = self._hf_upload_manager.should_upload(self.global_steps)

                if should_save_ckpt or should_upload_hf:
                    self._hf_upload_manager.flush()
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                if should_upload_hf:
                    self._hf_upload_manager.maybe_upload(self.global_steps)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger_tracker.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._flush_image_dumps()
                    self._hf_upload_manager.flush()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
