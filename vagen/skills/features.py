from abc import ABC, abstractmethod
from typing import List

import numpy as np


class FeatureExtractor(ABC):

    @abstractmethod
    def extract(self, env_state: dict) -> List[int]:
        """Extract structural feature vector from raw env state."""

    @abstractmethod
    def similarity(self, query: List[int], target: List[int]) -> float:
        """Compute similarity between two feature vectors. Returns 0.0~1.0."""


class SokobanFeatureExtractor(FeatureExtractor):
    """
    Per-box 12-dim features:
      [0:2]  player→box direction sign (row, col)
      [2:4]  player→box distance (row, col)
      [4:6]  box→nearest_target direction sign (row, col)
      [6:8]  box→nearest_target distance (row, col)
      [8:12] walls around box (up, down, left, right)
    """

    DIMS_PER_BOX = 12

    def extract(self, env_state: dict) -> List[int]:
        room_state = env_state["room_state"]
        room_fixed = env_state["room_fixed"]
        walls = room_fixed == 0
        targets = room_fixed == 2
        boxes = (room_state == 4) | (room_state == 3)
        player = np.argwhere(room_state == 5)[0]
        box_pos = np.argwhere(boxes)
        target_pos = np.argwhere(targets)
        rows, cols = room_state.shape

        features = []
        for bp in box_pos:
            pr, pc = np.sign(player - bp)
            features.extend([int(pr), int(pc)])
            features.extend([int(abs(player[0] - bp[0])), int(abs(player[1] - bp[1]))])

            dists = np.abs(target_pos - bp).sum(axis=1)
            nearest_t = target_pos[dists.argmin()]
            tr, tc = np.sign(bp - nearest_t)
            features.extend([int(tr), int(tc)])
            features.extend([int(abs(bp[0] - nearest_t[0])), int(abs(bp[1] - nearest_t[1]))])

            r, c = bp
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                blocked = int(
                    nr < 0 or nr >= rows or nc < 0 or nc >= cols or walls[nr, nc]
                )
                features.append(blocked)

        return features

    def similarity(
        self,
        query: List[int],
        target: List[int],
        sign_weight: float = 0.4,
        dist_weight: float = 0.4,
        wall_weight: float = 0.2,
    ) -> float:
        a, b = np.array(query), np.array(target)
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        n_boxes = len(a) // self.DIMS_PER_BOX
        sign_scores, dist_scores, wall_scores = [], [], []

        for i in range(n_boxes):
            base = i * self.DIMS_PER_BOX
            sign_scores.append(
                float(
                    (a[base : base + 2] == b[base : base + 2]).mean()
                    + (a[base + 4 : base + 6] == b[base + 4 : base + 6]).mean()
                )
                / 2.0
            )
            dist_scores.append(
                float(
                    (a[base + 2 : base + 4] == b[base + 2 : base + 4]).mean()
                    + (a[base + 6 : base + 8] == b[base + 6 : base + 8]).mean()
                )
                / 2.0
            )
            wall_scores.append(
                float((a[base + 8 : base + 12] == b[base + 8 : base + 12]).mean())
            )

        return float(
            sign_weight * np.mean(sign_scores)
            + dist_weight * np.mean(dist_scores)
            + wall_weight * np.mean(wall_scores)
        )


class FrozenLakeFeatureExtractor(FeatureExtractor):
    """
    10-dim features:
      [0:2]  S→G orientation sign (row, col)
      [2:4]  S→G distance (row, col)
      [4:10] hole distribution (S row-adj, S col-adj, S diag,
                                G row-adj, G col-adj, G diag)
    """

    def extract(self, env_state: dict) -> List[int]:
        desc = env_state["desc"]
        nrow, ncol = len(desc), len(desc[0])
        start = goal = None
        for r in range(nrow):
            for c in range(ncol):
                if desc[r][c] == "S":
                    start = (r, c)
                elif desc[r][c] == "G":
                    goal = (r, c)

        sr, sc = start
        gr, gc = goal
        sign_r, sign_c = int(np.sign(gr - sr)), int(np.sign(gc - sc))

        def _is_hole(r, c):
            if 0 <= r < nrow and 0 <= c < ncol:
                return int(desc[r][c] == "H")
            return 0

        features = []
        # sign (2 dims)
        features.extend([sign_r, sign_c])
        # dist (2 dims)
        features.extend([abs(gr - sr), abs(gc - sc)])
        # hole distribution (6 dims): S side (row, col, diag) + G side (row, col, diag)
        features.append(_is_hole(sr + sign_r, sc))           # S row-adj toward G
        features.append(_is_hole(sr, sc + sign_c))           # S col-adj toward G
        features.append(_is_hole(sr + sign_r, sc + sign_c))  # S diag toward G
        features.append(_is_hole(gr - sign_r, gc))           # G row-adj toward S
        features.append(_is_hole(gr, gc - sign_c))           # G col-adj toward S
        features.append(_is_hole(gr - sign_r, gc - sign_c))  # G diag toward S

        return features

    def similarity(
        self,
        query: List[int],
        target: List[int],
        sign_weight: float = 0.25,
        dist_weight: float = 0.25,
        hole_weight: float = 0.5,
    ) -> float:
        a, b = np.array(query), np.array(target)
        if len(a) != 10 or len(b) != 10:
            return 0.0

        sign_sim = float((a[0:2] == b[0:2]).mean())
        dist_sim = float((a[2:4] == b[2:4]).mean())
        hole_sim = float((a[4:10] == b[4:10]).mean())

        return float(
            sign_weight * sign_sim + dist_weight * dist_sim + hole_weight * hole_sim
        )


class ManiSkillFeatureExtractor(FeatureExtractor):
    """
    Hard-partitioned tabletop layout features:
      [0:2N]  millimeter xy positions of task-relevant movable objects
      [2N]    task index (skills never transfer across tasks)
    """

    DISTANCE_SCALE = 400.0

    def extract(self, env_state: dict) -> List[int]:
        from vagen.envs.primitive_skill.maniskill.oracle import (
            TASK_INDEX,
            extract_structural_features,
        )
        env_id = env_state.get("env_id", "")
        initial_state = env_state.get("initial_state")
        if env_id not in TASK_INDEX or not initial_state:
            return []
        features = extract_structural_features(env_id, initial_state)
        features.append(TASK_INDEX[env_id])
        return features

    def similarity(self, query: List[int], target: List[int]) -> float:
        a = np.asarray(query, dtype=float)
        b = np.asarray(target, dtype=float)
        if a.shape != b.shape or a.size < 3 or a.size % 2 == 0:
            return 0.0
        if a[-1] != b[-1]:  # different task
            return 0.0
        distances = np.linalg.norm(
            a[:-1].reshape(-1, 2) - b[:-1].reshape(-1, 2), axis=1
        )
        return max(0.0, 1.0 - float(distances.max()) / self.DISTANCE_SCALE)


EXTRACTOR_REGISTRY = {
    "sokoban": SokobanFeatureExtractor,
    "frozenlake": FrozenLakeFeatureExtractor,
    "maniskill": ManiSkillFeatureExtractor,
}


def get_feature_extractor(env_type: str) -> FeatureExtractor:
    cls = EXTRACTOR_REGISTRY.get(env_type)
    if cls is None:
        raise ValueError(
            f"Unknown env_type '{env_type}', available: {list(EXTRACTOR_REGISTRY)}"
        )
    return cls()
