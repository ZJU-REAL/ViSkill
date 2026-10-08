import json
import math
import os
from typing import List, Optional

from PIL import Image

from .features import FeatureExtractor


class ViSkillLibrary:

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        max_size: int = 32,
        ema_alpha: float = 0.1,
        utility_initial: float = 0.5,
        similarity_weight: float = 0.9,
    ):
        self.feature_extractor = feature_extractor
        self.max_size = max_size
        self.ema_alpha = ema_alpha
        self.utility_initial = utility_initial
        self.similarity_weight = similarity_weight
        self._skills: List[dict] = []
        self._next_id = 0
        self._pending_images: dict = {}

    def __len__(self) -> int:
        return len(self._skills)

    def avg_utility(self) -> float:
        if not self._skills:
            return 0.0
        return sum(s["utility"] for s in self._skills) / len(self._skills)

    def _score(self, query: List[int], skill: dict) -> float:
        similarity = self.feature_extractor.similarity(query, skill["features"])
        return (
            self.similarity_weight * similarity
            + (1 - self.similarity_weight) * skill["utility"]
        )

    def retrieve(self, env_state: dict) -> Optional[dict]:
        if not self._skills:
            return None
        query = self.feature_extractor.extract(env_state)
        best_score = -1.0
        best_skill = None
        for skill in self._skills:
            score = self._score(query, skill)
            if score > best_score:
                best_score = score
                best_skill = skill
        if best_skill is None:
            return None
        return {**best_skill, "score": best_score}

    def admit(self, skill_data: dict) -> Optional[str]:
        """Admit a new skill. Evicts lowest-score skill if library is full."""
        if len(self._skills) >= self.max_size:
            self._evict()

        skill_id = f"skill_{self._next_id}"
        self._next_id += 1

        image = skill_data.get("image")
        image_path = ""
        env_id = skill_data.get("env_id")
        if image is not None:
            image_path = f"{env_id}/{skill_id}.png" if env_id else f"{skill_id}.png"
            self._pending_images[skill_id] = (image, image_path)

        skill = {
            "id": skill_id,
            "seed": skill_data.get("seed"),
            "env_id": env_id,
            "features": skill_data["features"],
            "strategy": skill_data.get("strategy", ""),
            "image_path": image_path,
            "utility": self.utility_initial,
            "use_count": 0,
        }
        if "trajectory" in skill_data:
            skill["trajectory"] = skill_data["trajectory"]
        self._skills.append(skill)
        return skill_id

    def update_utility(self, skill_id: str, reward: float) -> None:
        for skill in self._skills:
            if skill["id"] == skill_id:
                skill["utility"] = (1 - self.ema_alpha) * skill["utility"] + self.ema_alpha * reward
                skill["use_count"] += 1
                return

    def load_image(self, skill: dict, library_path: str) -> Optional[Image.Image]:
        path = skill.get("image_path", "")
        if not path:
            return None
        full = os.path.join(library_path, path)
        if os.path.exists(full):
            return Image.open(full).convert("RGB")
        return None

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        for _, (img, rel_path) in self._pending_images.items():
            full_path = os.path.join(path, rel_path)
            dir_name = os.path.dirname(full_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            img.save(full_path)
        self._pending_images.clear()

        valid_paths = set()
        index = []
        for skill in self._skills:
            entry = {
                "id": skill["id"],
                "seed": skill.get("seed"),
                "env_id": skill.get("env_id"),
                "features": skill["features"],
                "strategy": skill.get("strategy", ""),
                "image_path": skill["image_path"],
                "utility": skill["utility"],
                "use_count": skill["use_count"],
            }
            if "trajectory" in skill:
                entry["trajectory"] = skill["trajectory"]
            index.append(entry)
            if skill["image_path"]:
                valid_paths.add(skill["image_path"])

        with open(os.path.join(path, "skill_index.json"), "w") as f:
            json.dump({"next_id": self._next_id, "skills": index}, f, indent=2)

        for root, _, files in os.walk(path):
            for fname in files:
                if not fname.endswith(".png"):
                    continue
                full_path = os.path.join(root, fname)
                if os.path.relpath(full_path, path) not in valid_paths:
                    os.remove(full_path)

    def load(self, path: str) -> None:
        index_path = os.path.join(path, "skill_index.json")
        if not os.path.exists(index_path):
            return
        with open(index_path) as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 0)
        self._skills = data.get("skills", [])

    def _remove(self, index: int) -> None:
        evicted = self._skills.pop(index)
        self._pending_images.pop(evicted["id"], None)

    def _evict(self) -> None:
        if not self._skills:
            return
        self._skills.sort(
            key=lambda s: s["utility"] * math.log2(s["use_count"] + 1),
            reverse=True,
        )
        self._remove(len(self._skills) - 1)
