"""
Prompt templates for the primitive_skill (ManiSkill robot manipulation) environment.

Structure:
  - system_prompt(): shared base prompt (robot description, action space, hardcoded example)
  - format_prompt(): format instruction + optional XML-tagged examples (dispatches by prompt_format)
  - init_observation_template(): first observation
  - action_template(): subsequent observations
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base system prompt (robot description + action space + multi-round example)
# ---------------------------------------------------------------------------

def system_prompt() -> str:
    """Return the base system prompt for the primitive skill task."""
    return _ROBOT_DESCRIPTION


_ROBOT_DESCRIPTION = """\
You are an AI assistant controlling a Franka Emika robot arm. Your goal is to understand human instructions and translate them into a sequence of executable actions for the robot, based on visual input and the instruction.

Action Space Guide
You can command the robot using the following actions:

1. pick(x, y, z) # To grasp an object located at position(x,y,z) in the robot's workspace.
2. place(x, y, z) # To place the object currently held by the robot's gripper at the target position (x,y,z).
3. push(x1, y1, z1, x2, y2, z2) # To push an object from position (x1,y1,z1) to (x2,y2,z2).

Hints:
1. The coordinates (x, y, z) are in millimeters and are all integers.
2. Please ensure that the coordinates are within the workspace limits.
3. The position is the center of the object, when you place, please consider the volume of the object. It's always fine to set z much higher when placing an item.
4. We will provide the object positions to you, but you need to match them to the object in the image by yourself. You're facing toward the negative x-axis, and the negative y-axis is to your left, the positive y-axis is to your right, and the positive z-axis is up.

Examples:
round1:
image1
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(62,-55,20),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
Reasoning: I can see from the picture that the red cube is on my left and green cube is on my right and near me.
Since I'm looking toward the negative x axis, and negative y-axis is to my left, (62,-55,20) would be the position of the red cube, (75,33,20) would be the position of the green cube and (-44,100,20) is the position of the yellow cube.
Also the (100,-43,0) would be the position of the left target, and (100,43,0) would be the porition of the right target.
I need to pick up red cube first and place it on the green cube, when placing, I should set z much higher.
Anwer: pick(62,-55,20)|place(75,33,50)
round2:
image2
Human Instruction: Put red cube on green cube and yellow cube on left target
Object positions:
[(75,33,50),(75,33,20),(-44,100,20),(100,-43,0),(100,43,0)]
Reasoning: Now the red cube is on the green cube, so I need to pick up the yellow cube and place it on the left target.
Anwer: pick(-44,100,20)|place(100,-43,50)"""


# ---------------------------------------------------------------------------
# Format prompt (format instruction + optional examples)
# ---------------------------------------------------------------------------

def format_prompt(max_actions_per_step: int = 2, action_sep: str = "|",
                  add_example: bool = True, prompt_format: str = "free_think") -> str:
    """Generate format prompt based on the specified format.

    Args:
        max_actions_per_step: max actions per turn.
        action_sep: separator between actions.
        add_example: whether to append few-shot examples.
        prompt_format: one of free_think, wm.
    """
    if prompt_format == "free_think":
        return _free_think_format_prompt(max_actions_per_step, action_sep, add_example)
    elif prompt_format == "wm":
        return _wm_format_prompt(max_actions_per_step, action_sep, add_example)
    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")


VALID_FORMATS = ["free_think", "wm"]


def _free_think_format_prompt(max_actions_per_step: int, action_sep: str, add_example: bool) -> str:
    base_prompt = (
        f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.\n"
        f"You should first give your thought process, and then your answer.\n"
        f"Your response should be in the format of:\n"
        f"<think>...</think><answer>...</answer>"
    )
    if add_example:
        examples = f"""
Example:
<think>I need to pick the red cube at (62,-55,20) and place it on the green cube at (75,33,20). \
When placing, I should set z much higher to stack properly.</think>
<answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>"""
        return base_prompt + "\n" + examples
    return base_prompt


def _wm_format_prompt(max_actions_per_step: int, action_sep: str, add_example: bool) -> str:
    base_prompt = (
        f"You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.\n"
        f"You need to describe your observation, think, give your answer, then predict "
        f"what you will see next.\n"
        f"Your response should be in the format of:\n"
        f"<observation>...</observation><think>...</think><answer>...</answer><prediction>...</prediction>"
    )
    if add_example:
        examples = f"""
Example:
<observation>I can see the red cube on my left and green cube on my right. \
The red cube is at (62,-55,20), green cube at (75,33,20), yellow cube at (-44,100,20).</observation>
<think>I need to pick up the red cube first and place it on the green cube. \
When placing, I should set z much higher.</think>
<answer>pick(62,-55,20){action_sep}place(75,33,50)</answer>
<prediction>After this action, the red cube will be stacked on top of the green cube \
at approximately (75,33,50).</prediction>"""
        return base_prompt + "\n" + examples
    return base_prompt


# ---------------------------------------------------------------------------
# Observation templates
# ---------------------------------------------------------------------------

def init_observation_template(
    observation: str = "<image>",
    instruction: str = "",
    x_workspace: tuple = (0, 0),
    y_workspace: tuple = (0, 0),
    z_workspace: tuple = (0, 0),
    object_positions: str = "",
    other_information: str = "",
    **kwargs,
) -> str:
    return f"""
[Initial Observation]:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}
Object positions:
{object_positions}
Other information:
{other_information}
Decide your next action(s)."""


def action_template(
    valid_actions: list = None,
    observation: str = "<image>",
    instruction: str = "",
    x_workspace: tuple = (0, 0),
    y_workspace: tuple = (0, 0),
    z_workspace: tuple = (0, 0),
    object_positions: str = "",
    other_information: str = "",
    **kwargs,
) -> str:
    return f"""After your answer, the extracted valid action(s) is {valid_actions}.
After that, the observation is:
{observation}
Human Instruction: {instruction}
x_workspace_limit: {x_workspace}
y_workspace_limit: {y_workspace}
z_workspace_limit: {z_workspace}
Object positions:
{object_positions}
Other information:
{other_information}
Decide your next action(s)."""
