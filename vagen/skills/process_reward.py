"""Per-step process reward based on BFS distance change."""

from typing import Optional

from .solver import get_solver


class ProcessRewardComputer:

    def __init__(self, env_type: str):
        self._solver = get_solver(env_type)

    def distance_to_goal(self, env_state: dict) -> Optional[int]:
        return self._solver.distance(env_state)

    def compute(
        self,
        state_before: dict,
        state_after: dict,
        step_reward: float = 0.05,
        unsolvable_penalty: float = -0.2,
    ) -> float:
        dist_before = self.distance_to_goal(state_before)
        dist_after = self.distance_to_goal(state_after)

        if dist_before is None:
            return 0.0
        if dist_after is None:
            return unsolvable_penalty

        delta = dist_before - dist_after
        if delta > 0:
            return step_reward
        elif delta < 0:
            return -step_reward
        return 0.0


def get_process_reward_computer(env_type: str) -> ProcessRewardComputer:
    return ProcessRewardComputer(env_type)
