"""Distillation prompt construction (shared by self-distill and api-distill)."""

from typing import List, Optional
from PIL import Image


DISTILL_TEMPLATE = (
    "[Skill Distillation]\n"
    "Task: {task_description}\n\n"
    "You successfully completed this task. Based on the initial state and your trajectory, "
    "distill a reusable strategy that can guide solving similar tasks.\n\n"
    "The strategy should describe WHEN to apply it (what the state looks like) "
    "and WHAT to do (concrete actions or approach).\n\n"
    "Your response should be in the format of:\n"
    "<strategy>...</strategy>\n\n"
    "{examples}"
    "Initial state:\n<image>\n"
    "{trajectory_section}\n"
    "Now distill your strategy:"
)

DISTILL_EXAMPLES = {
    "sokoban": (
        "Example 1:\n"
        "<strategy>When the box is directly between the player and the target in a straight line, "
        "push the box toward the target repeatedly until it reaches the goal.</strategy>\n\n"
        "Example 2:\n"
        "<strategy>When the target requires pushing the box around a corner, "
        "first reposition to push the box in one direction, "
        "then move around the box to push it in the new direction toward the target.</strategy>\n\n"
    ),
    "frozenlake": (
        "Example 1:\n"
        "<strategy>When the goal is directly adjacent with no holes in between, "
        "move straight toward it.</strategy>\n\n"
        "Example 2:\n"
        "<strategy>When holes block the direct path to the goal, "
        "first move sideways to a clear column or row, "
        "then navigate toward the goal along frozen tiles.</strategy>\n\n"
    ),
    "maniskill": {
        "AlignTwoCube": (
            "Example 1:\n"
            "<strategy>When two cubes must be aligned on the y-axis, pick each "
            "cube and place it at x=0 while keeping its y position.</strategy>\n\n"
        ),
        "PlaceTwoCube": (
            "Example 1:\n"
            "<strategy>When each cube has its own target, place the red cube "
            "on the left target first, then place the green cube on the right "
            "target.</strategy>\n\n"
        ),
        "StackThreeCube": (
            "Example 1:\n"
            "<strategy>When stacking three cubes, first place the red cube on "
            "top of the green cube, then place the purple cube on top of the "
            "red cube.</strategy>\n\n"
        ),
        "PutAppleInDrawer": (
            "Example 1:\n"
            "<strategy>When the drawer is already open, pick the apple, place "
            "it inside the drawer, then push the handle until the drawer "
            "closes.</strategy>\n\n"
        ),
        "SwapTwoCube": (
            "Example 1:\n"
            "<strategy>When swapping two cubes, first move one cube to the "
            "middle buffer zone, then place the other cube on its target, and "
            "finally move the buffered cube to the remaining target.</strategy>\n\n"
        ),
    },
}


def build_distill_prompt(
    task_description: str,
    actions: List[str],
    env_type: str = "",
    has_trajectory_image: bool = True,
    env_id: str = "",
) -> str:
    examples = DISTILL_EXAMPLES.get(env_type, "")
    if isinstance(examples, dict):
        examples = examples.get(env_id, "")
    if has_trajectory_image:
        trajectory_section = "Successful trajectory:\n<image>\n"
    else:
        action_str = ", ".join(actions)
        trajectory_section = f"Actions taken: {action_str}\n"
    return DISTILL_TEMPLATE.format(
        task_description=task_description,
        examples=examples,
        trajectory_section=trajectory_section,
    )


def build_distill_images(
    init_frame: Optional[Image.Image],
    trajectory_image: Optional[Image.Image],
) -> List[Image.Image]:
    images = []
    if init_frame:
        images.append(init_frame)
    if trajectory_image:
        images.append(trajectory_image)
    return images
