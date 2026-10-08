from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class SkillProvider(ABC):

    @abstractmethod
    def retrieve(self, env_state: dict) -> Optional[dict]:
        """Return the most similar skill to the current env state, or None."""

    @abstractmethod
    def format_for_prompt(self, skill: dict) -> Dict[str, Any]:
        """Format a skill into VAGEN obs protocol: {obs_str, multi_modal_input}."""

    @abstractmethod
    def update_utility(self, skill_id: str, reward: float) -> None:
        """EMA-update utility score for a retrieved skill after an episode."""

    @abstractmethod
    def compute_process_reward(
        self, state_before: dict, state_after: dict
    ) -> float:
        """Per-step progress reward based on state transition."""

    @abstractmethod
    def save(self) -> None:
        """Persist the skill library to disk."""
