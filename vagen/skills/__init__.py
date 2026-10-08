from .base import SkillProvider
from .provider import ViSkillProvider


def get_skill_provider(config: dict) -> SkillProvider:
    return ViSkillProvider(config)
