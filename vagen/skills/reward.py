"""Reward computation for skill-guided training."""

from typing import List, Optional


def compute_prefix_reward(skill_trajectory: List[str], model_trajectory: List[str]) -> float:
    """Reward based on how closely model followed skill's trajectory (prefix matching)."""
    if not skill_trajectory:
        return 0.0
    prefix_len = 0
    for s_act, m_act in zip(skill_trajectory, model_trajectory):
        if s_act == m_act:
            prefix_len += 1
        else:
            break
    extra_steps = max(0, len(model_trajectory) - prefix_len)
    return max(0.0, (prefix_len - extra_steps) / len(skill_trajectory))


def compute_reward(
    env_rewards: List[float],
    mode: str,
    traj_success: bool,
    skill_trajectory: Optional[List[str]] = None,
    model_trajectory: Optional[List[str]] = None,
    guided_weight: float = 0.5,
) -> float:
    """Compute final reward based on mode.

    env_rewards already includes per-step signals from the environment:
      - format_reward (valid action bonus)
      - success_reward (task completion)
      - process_reward (BFS distance change, only when mode=bfs_process)

    This function adds mode-specific bonuses on top:
      success:      env_rewards only
      bfs_process:  env_rewards only (process reward already accumulated per-step)
      skill_guided: env_rewards + weighted prefix match bonus on failure
    """
    base = sum(env_rewards) if env_rewards else 0.0

    if mode == "skill_guided" and not traj_success:
        if skill_trajectory and model_trajectory:
            return base + guided_weight * compute_prefix_reward(skill_trajectory, model_trajectory)

    return base
