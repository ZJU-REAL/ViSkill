"""
Generate cold-start skill library via BFS with diversity sampling.

Supports local envs (sokoban, frozenlake) and ManiSkill primitive-skill
envs (maniskill).
ManiSkill uses scripted oracle planners and samples skills per task.

Usage:
  python -m vagen.skills.cold_start \
      --config-path=examples/train/sokoban \
      --config-name=cold_start_sokoban
"""

import json
import os
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from vagen.skills.features import get_feature_extractor
from vagen.skills.solver import get_solver
from vagen.skills.renderer import (
    get_skill_renderer,
    compose_skill_image,
    render_frames_with_path,
    render_config_options,
)
from vagen.skills.api_distiller import ViSkillDistiller
from vagen.skills.provider import (
    TASK_DESCRIPTION_OVERRIDES,
    TASK_DESCRIPTIONS,
    append_strategy_suffix,
)


def _create_sokoban_env(seed, env_config):
    from vagen.envs.sokoban.patch_sokoban_env import PatchedSokobanEnv
    env = PatchedSokobanEnv(
        dim_room=tuple(env_config["dim_room"]),
        max_steps=100,
        num_boxes=env_config.get("num_boxes", 1),
    )
    env.reset(
        seed=seed,
        min_solution_steps=env_config.get("min_solution_steps"),
        min_solution_bfs_max_depth=env_config.get("bfs_max_depth", 200),
    )
    return {"room_state": env.room_state.copy(), "room_fixed": env.room_fixed.copy()}


def _create_frozenlake_env(seed, env_config):
    from vagen.envs.frozenlake.utils.utils import generate_random_map
    size = env_config.get("size", 4)
    p = env_config.get("p", 0.8)
    desc = generate_random_map(size=size, p=p, seed=seed)
    return {"desc": [list(row) for row in desc]}


ENV_RESET_REGISTRY = {
    "sokoban": _create_sokoban_env,
    "frozenlake": _create_frozenlake_env,
}


# ---------------------------------------------------------------------------
# ManiSkill: scripted oracle planners (one PrimitiveSkillEnv per task)
# ---------------------------------------------------------------------------


def _maniskill_solve_seed(env, env_id: str, seed: int) -> Optional[dict]:
    """Run the scripted oracle on one seed. Returns candidate data."""
    from vagen.envs.primitive_skill.maniskill.oracle import run_oracle_episode
    try:
        result = run_oracle_episode(env, env_id, seed)
    except Exception:
        return None
    if result is None:
        return None
    return {
        "frames": result["frames"],
        "actions": result["plan"]["actions"],
        "strategy": result["plan"]["strategy"],
        "seed": seed,
        "env_id": env_id,
        "init_state": {"env_id": env_id, "initial_state": result["initial_positions"]},
    }


def solve_seed(seed, env_type, env_config, solver) -> Optional[dict]:
    """Reset env with seed, solve via solver, collect trajectory."""
    reset_fn = ENV_RESET_REGISTRY.get(env_type)
    if reset_fn is None:
        return None
    try:
        env_state = reset_fn(seed, env_config)
    except Exception:
        return None

    result = solver.replay(env_state)
    if result is None:
        return None

    min_steps = env_config.get("min_solution_steps")
    if min_steps and not (min_steps[0] <= len(result["actions"]) <= min_steps[1]):
        return None

    result["seed"] = seed
    result["init_state"] = env_state
    return result


def farthest_point_sampling(
    candidates: list, extractor, size: int, dedup_threshold: float = 0.85
) -> list:
    """Select structurally extreme candidates with hard deduplication.

    This intentionally tracks each candidate's minimum similarity to the
    selected set and favors the smallest value. Unlike standard max-min FPS,
    the objective expands feature-space span by finding a sample that is very
    different from at least one selected example. ``dedup_threshold`` remains
    the hard constraint that prevents similarity to any selected example from
    becoming too high.
    """
    if len(candidates) <= size:
        return candidates

    n = len(candidates)
    selected = [0]
    selected_set = {0}
    excluded = set()
    min_sims = [float("inf")] * n

    for _ in range(size - 1):
        last = selected[-1]
        for j in range(n):
            if j in excluded or j in selected_set:
                continue
            sim = extractor.similarity(
                candidates[last]["features"], candidates[j]["features"]
            )
            min_sims[j] = min(min_sims[j], sim)
            if sim > dedup_threshold:
                excluded.add(j)

        pool = [j for j in range(n) if j not in excluded and j not in selected_set]
        if not pool:
            break
        best = min(pool, key=lambda j: min_sims[j])
        selected.append(best)
        selected_set.add(best)

    return [candidates[i] for i in selected]


def render_skill(frames, actions, strategy, render_options):
    """Compose skill image with optional strategy text."""
    strategy_in_image = render_options["strategy_enable"]
    trajectory_enable = render_options["trajectory_enable"]

    if not trajectory_enable and not strategy_in_image:
        return None

    return compose_skill_image(
        frames, actions,
        strategy_text=strategy if strategy_in_image else "",
        thumbnail_size=render_options["thumbnail_size"],
        show_title=render_options["show_title"],
        trajectory_enable=trajectory_enable,
        strategy_enable=strategy_in_image,
        action_type=render_options["action_type"],
        scale=render_options["scale"],
    )


@hydra.main(version_base=None)
def main(config: DictConfig):
    cfg = OmegaConf.to_container(config, resolve=True)

    env_type = cfg["env"]
    seed_start, seed_end = cfg["seed"][0], cfg["seed"][1]
    sizes = cfg["size"] if isinstance(cfg["size"], list) else [cfg["size"]]
    max_size = max(sizes)
    dedup_threshold = cfg.get("dedup_threshold", 0.85)
    output_base = cfg.get("output") or f"./cold_start_skill_library/{env_type}"
    env_config = cfg["env_config"]
    render_cfg = cfg.get("render", {})
    render_options = render_config_options(render_cfg)
    distill_cfg = cfg.get("distill", {})

    extractor = get_feature_extractor(env_type)
    solver = get_solver(env_type)
    try:
        renderer = get_skill_renderer(env_type)
    except ValueError:
        renderer = None  # maniskill composes skill images without a renderer
    draw_path = render_options["draw_path"]
    is_maniskill = (env_type == "maniskill")

    # ManiSkill: one PrimitiveSkillEnv per task (GPU bound at first reset)
    maniskill_envs = {}
    if is_maniskill:
        from vagen.envs.primitive_skill.primitive_skill_env import PrimitiveSkillEnv
        from vagen.envs.primitive_skill.maniskill.oracle import ALL_ENV_IDS
        gpu = env_config.get("gpu")
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env_ids = env_config.get("env_ids") or list(ALL_ENV_IDS)
        maniskill_envs = {
            env_id: PrimitiveSkillEnv({
                "env_id": env_id,
                "render_mode": "vision",
                "prompt_format": "free_think",
                "max_actions_per_step": 1,
                "max_steps": env_config.get("max_steps", 10),
            })
            for env_id in env_ids
        }

    distiller = None
    if distill_cfg.get("model"):
        distiller = ViSkillDistiller(
            model=distill_cfg["model"],
            api_base=distill_cfg.get("api_base"),
            api_key=distill_cfg.get("api_key"),
            max_retries=distill_cfg.get("max_retries", 3),
        )

    # Phase 1: Generate candidates
    print(f"Scanning seeds {seed_start}~{seed_end}...")
    candidates = []
    progress_interval = 1 if seed_end - seed_start <= 20 else 50
    try:
        for seed in range(seed_start, seed_end):
            if (seed - seed_start) % progress_interval == 0:
                print(f"  Processing seed {seed}...", flush=True)
            if is_maniskill:
                results = [
                    _maniskill_solve_seed(menv, env_id, seed)
                    for env_id, menv in maniskill_envs.items()
                ]
            else:
                results = [solve_seed(seed, env_type, env_config, solver)]
            for result in results:
                if result is None:
                    continue
                result["features"] = extractor.extract(result["init_state"])
                candidates.append(result)
                if len(candidates) % 500 == 0:
                    print(f"  {len(candidates)} candidates found...")
    finally:
        for menv in maniskill_envs.values():
            menv._sync_close()

    print(f"Found {len(candidates)} solvable seeds")
    if not candidates:
        print("No candidates found, exiting.")
        return

    # Phase 2: Diversity sampling (run once with max size)
    if is_maniskill:
        # Per-task sampling: cross-task similarity is always 0 by design
        selected = []
        for env_id in maniskill_envs:
            group = [c for c in candidates if c["env_id"] == env_id]
            selected.extend(
                farthest_point_sampling(group, extractor, max_size, dedup_threshold)
            )
    else:
        selected = farthest_point_sampling(candidates, extractor, max_size, dedup_threshold)
    print(f"Selected {len(selected)} diverse seeds")

    # Phase 3: Render, distill strategy, and save
    all_skills = []
    task_desc = TASK_DESCRIPTIONS.get(env_type, "")

    for i, c in enumerate(selected):
        skill_id = f"skill_{i}"

        frames = (
            render_frames_with_path(c["frames"], c["positions"], renderer)
            if draw_path and "positions" in c
            else c["frames"]
        )
        init_frame = c["frames"][0]
        case_desc = TASK_DESCRIPTION_OVERRIDES.get(c.get("env_id"), task_desc)
        strategy = ""
        if distiller:
            traj_image = render_skill(frames, c["actions"], "", render_options)
            result = distiller.distill({
                "trajectory": c["actions"],
                "init_frame": init_frame,
                "trajectory_image": traj_image,
                "task_description": case_desc,
                "env_type": env_type,
                "env_id": c.get("env_id", ""),
            })
            strategy = result.get("strategy", "")
            print(f"  [{i+1}/{len(selected)}] seed={c['seed']}, strategy={strategy[:60]}...")
        else:
            strategy = c.get("strategy") or "BFS optimal: " + " → ".join(c["actions"])
            print(f"  [{i+1}/{len(selected)}] seed={c['seed']}")
        strategy = append_strategy_suffix(env_type, strategy)
        image = render_skill(frames, c["actions"], strategy, render_options)

        image_path = ""
        if image is not None:
            prefix = f"{c['env_id']}/" if is_maniskill else ""
            image_path = f"{prefix}{skill_id}.png"
        skill = {
            "id": skill_id,
            "seed": c["seed"],
            "features": c["features"],
            "strategy": strategy,
            "image_path": image_path,
            "image": image,
            "utility": 0.5,
            "use_count": 0,
        }
        skill["trajectory"] = c["actions"]
        if is_maniskill:
            skill["env_id"] = c["env_id"]
        all_skills.append(skill)

    # Phase 4: Save libraries for each size (prefix slicing)
    for size in sorted(sizes):
        output_dir = f"{output_base}_lib{size}" if len(sizes) > 1 else output_base
        if is_maniskill:
            # Slice per task so every task keeps `size` skills
            subset = [
                skill
                for env_id in maniskill_envs
                for skill in [s for s in all_skills if s.get("env_id") == env_id][:size]
            ]
        else:
            subset = all_skills[:size]
        os.makedirs(output_dir, exist_ok=True)

        for s in subset:
            if s["image"] is not None:
                full_path = os.path.join(output_dir, s["image_path"])
                dir_name = os.path.dirname(full_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                s["image"].save(full_path)

        index = {"next_id": len(subset), "skills": [
            {k: v for k, v in s.items() if k != "image"} for s in subset
        ]}
        with open(os.path.join(output_dir, "skill_index.json"), "w") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(subset)} skills to {output_dir}")


if __name__ == "__main__":
    main()
