"""Environment solvers: BFS solve + replay to collect trajectory data."""

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional

import numpy as np
from PIL import Image


class EnvSolver(ABC):

    @abstractmethod
    def solve(self, env_state: dict) -> Optional[List[int]]:
        """Return action id sequence or None if unsolvable."""

    @abstractmethod
    def replay(self, env_state: dict) -> Optional[dict]:
        """Solve + replay, return {frames, actions, positions} or None."""

    def distance(self, env_state: dict) -> Optional[int]:
        """Return shortest distance (number of actions) to goal."""
        path = self.solve(env_state)
        return len(path) if path is not None else None


class SokobanSolver(EnvSolver):

    ACTION_ID_TO_NAME = {1: "up", 2: "down", 3: "left", 4: "right"}

    def __init__(self, dim_room=(6, 6), num_boxes=1, max_steps=100, bfs_max_depth=200):
        self.dim_room = dim_room
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        self.bfs_max_depth = bfs_max_depth

    def solve(self, env_state: dict) -> Optional[List[int]]:
        from vagen.envs.sokoban.patch_sokoban_env import get_shortest_action_path
        room_fixed = env_state.get("room_fixed")
        room_state = env_state.get("room_state")
        if room_fixed is None or room_state is None:
            return None
        path = get_shortest_action_path(room_fixed, room_state, MAX_DEPTH=self.bfs_max_depth)
        return path or None

    def replay(self, env_state: dict) -> Optional[dict]:
        from vagen.envs.sokoban.patch_sokoban_env import PatchedSokobanEnv

        action_seq = self.solve(env_state)
        if action_seq is None:
            return None

        room_state = env_state["room_state"]
        env = PatchedSokobanEnv(
            dim_room=self.dim_room, max_steps=self.max_steps, num_boxes=self.num_boxes
        )
        env.room_state = room_state.copy()
        env.room_fixed = env_state["room_fixed"].copy()
        env.player_position = np.argwhere(room_state == 5)[0]
        env.num_env_steps = 0
        env.boxes_on_target = int(np.sum((room_state == 3)))

        frames = [Image.fromarray(env.render("rgb_array"))]
        positions = [tuple(env.player_position)]
        actions = []

        for act_id in action_seq:
            env.step(act_id)
            positions.append(tuple(env.player_position))
            frames.append(Image.fromarray(env.render("rgb_array")))
            actions.append(self.ACTION_ID_TO_NAME[act_id])

        env.close()
        return {"frames": frames, "actions": actions, "positions": positions}


FROZENLAKE_DIRECTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


class FrozenLakeSolver(EnvSolver):

    ACTION_ID_TO_NAME = {0: "left", 1: "down", 2: "right", 3: "up"}

    @staticmethod
    def _find_cell(desc, char):
        for r in range(len(desc)):
            for c in range(len(desc[0])):
                if desc[r][c] == char:
                    return (r, c)
        return None

    def solve(self, env_state: dict) -> Optional[List[int]]:
        desc = env_state.get("desc")
        if desc is None:
            return None
        start = env_state.get("player_pos") or self._find_cell(desc, "S")
        goal = self._find_cell(desc, "G")
        if start is None or goal is None:
            return None
        if start == goal:
            return []

        nrow, ncol = len(desc), len(desc[0])
        visited = {start}
        queue = deque([(start, [])])

        while queue:
            (row, col), path = queue.popleft()
            for act_id, (dr, dc) in FROZENLAKE_DIRECTIONS.items():
                nr, nc = row + dr, col + dc
                if 0 <= nr < nrow and 0 <= nc < ncol and (nr, nc) not in visited and desc[nr][nc] != "H":
                    if (nr, nc) == goal:
                        return path + [act_id]
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [act_id]))
        return None

    def replay(self, env_state: dict) -> Optional[dict]:
        from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv

        action_seq = self.solve(env_state)
        if action_seq is None:
            return None

        desc = env_state["desc"]
        start = self._find_cell(desc, "S")

        env = GymFrozenLakeEnv(desc=desc, is_slippery=False, render_mode="rgb_array")
        env.reset()

        frames = [Image.fromarray(env.render())]
        positions = [start]
        actions = []

        for act_id in action_seq:
            dr, dc = FROZENLAKE_DIRECTIONS[act_id]
            positions.append((positions[-1][0] + dr, positions[-1][1] + dc))
            env.step(act_id)
            frames.append(Image.fromarray(env.render()))
            actions.append(self.ACTION_ID_TO_NAME[act_id])

        env.close()
        return {"frames": frames, "actions": actions, "positions": positions}


class ManiSkillSolver(EnvSolver):
    """Fixed optimal-step lookup for ManiSkill tasks.

    Each task has a fixed optimal action count (the scripted oracle plan
    length), used as the reference distance for distillation quality
    gating. solve/replay are unused: trajectories come from the env.
    """

    def solve(self, env_state: dict) -> Optional[List[int]]:
        return None

    def replay(self, env_state: dict) -> Optional[dict]:
        return None

    def distance(self, env_state: dict) -> Optional[int]:
        from vagen.envs.primitive_skill.maniskill.oracle import OPTIMAL_STEPS
        return OPTIMAL_STEPS.get(env_state.get("env_id", ""))


SOLVER_REGISTRY = {
    "sokoban": SokobanSolver,
    "frozenlake": FrozenLakeSolver,
    "maniskill": ManiSkillSolver,
}


def get_solver(env_type: str) -> EnvSolver:
    cls = SOLVER_REGISTRY.get(env_type)
    if cls is None:
        raise ValueError(f"Unknown env_type '{env_type}', available: {list(SOLVER_REGISTRY)}")
    return cls()
