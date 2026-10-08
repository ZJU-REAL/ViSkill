"""Skill trajectory renderers and shared image composition utilities."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def normalize_render_scale(scale: float = 1.0) -> float:
    """Validate and normalize the final skill-image scale."""
    try:
        scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"render.scale must be a number in (0, 1], got {scale!r}") from exc
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"render.scale must be in (0, 1], got {scale}")
    return scale


def resize_skill_image(image: Image.Image, scale: float = 1.0) -> Image.Image:
    """Resize a fully composed skill image while preserving its aspect ratio."""
    scale = normalize_render_scale(scale)
    if scale == 1.0:
        return image
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.LANCZOS)


class SkillRenderer(ABC):

    @abstractmethod
    def draw_trajectory_line(
        self, rgb_array: np.ndarray, positions: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Draw red path line on env rendering."""


class SokobanSkillRenderer(SkillRenderer):

    def __init__(self, dim_room: Tuple[int, int] = (6, 6)):
        self.dim_room = dim_room

    def draw_trajectory_line(
        self, rgb_array: np.ndarray, positions: List[Tuple[int, int]]
    ) -> np.ndarray:
        img = Image.fromarray(rgb_array)
        draw = ImageDraw.Draw(img)
        tile_h = img.height // self.dim_room[0]
        tile_w = img.width // self.dim_room[1]

        px = [(c * tile_w + tile_w // 2, r * tile_h + tile_h // 2) for r, c in positions]
        for i in range(1, len(px)):
            draw.line([px[i - 1], px[i]], fill="red", width=2)
        for x, y in px:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="red")
        return np.array(img)


class FrozenLakeSkillRenderer(SkillRenderer):

    def __init__(self, size: int = 4):
        self.size = size

    def draw_trajectory_line(
        self, rgb_array: np.ndarray, positions: List[Tuple[int, int]]
    ) -> np.ndarray:
        img = Image.fromarray(rgb_array)
        draw = ImageDraw.Draw(img)
        tile_h = img.height // self.size
        tile_w = img.width // self.size

        px = [(c * tile_w + tile_w // 2, r * tile_h + tile_h // 2) for r, c in positions]
        for i in range(1, len(px)):
            draw.line([px[i - 1], px[i]], fill="red", width=6)
        for x, y in px:
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill="red")
        return np.array(img)


RENDERER_REGISTRY = {
    "sokoban": SokobanSkillRenderer,
    "frozenlake": FrozenLakeSkillRenderer,
}


def get_skill_renderer(env_type: str) -> SkillRenderer:
    cls = RENDERER_REGISTRY.get(env_type)
    if cls is None:
        raise ValueError(
            f"Unknown env_type '{env_type}', available: {list(RENDERER_REGISTRY)}"
        )
    return cls()


def render_frames_with_path(
    frames: List[Image.Image],
    positions: List[Tuple[int, int]],
    renderer: SkillRenderer,
) -> List[Image.Image]:
    """Draw the cumulative path on each trajectory frame."""
    rendered = []
    for i, frame in enumerate(frames):
        array = renderer.draw_trajectory_line(np.array(frame), positions[:i + 1])
        rendered.append(Image.fromarray(array))
    return rendered


def normalize_render_config(render_cfg: dict) -> Tuple[dict, dict]:
    """Return normalized strategy and trajectory render sections."""
    strategy_cfg = render_cfg.get("strategy", {})
    if isinstance(strategy_cfg, bool):
        strategy_cfg = {"enable": strategy_cfg, "type": "vision"}
    trajectory_cfg = render_cfg.get("trajectory", {})
    if isinstance(trajectory_cfg, bool):
        trajectory_cfg = {"enable": trajectory_cfg}
    return strategy_cfg, trajectory_cfg


def render_config_options(render_cfg: dict) -> dict:
    """Build common keyword arguments for environment renderers."""
    strategy_cfg, trajectory_cfg = normalize_render_config(render_cfg)
    return {
        "strategy_enable": (
            strategy_cfg.get("enable", True)
            and strategy_cfg.get("type", "vision") == "vision"
        ),
        "trajectory_enable": trajectory_cfg.get("enable", True),
        "action_type": trajectory_cfg.get("action_type", "text"),
        "show_title": trajectory_cfg.get("show_title", True),
        "draw_path": trajectory_cfg.get("draw_path", False),
        "thumbnail_size": trajectory_cfg.get("thumbnail_size"),
        "scale": normalize_render_scale(render_cfg.get("scale", 1.0)),
    }


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.Draw) -> List[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _load_font(font_size: int):
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


ACTION_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


def compose_skill_image(
    trajectory_images: List[Image.Image],
    actions: List[str],
    strategy_text: str = "",
    thumbnail_size: Optional[int] = None,
    font_size: int = 15,
    show_title: bool = True,
    strategy_enable: bool = True,
    trajectory_enable: bool = True,
    action_type: str = "text",
    scale: float = 1.0,
) -> Image.Image:
    """Compose trajectory images into one composite image.

    trajectory_enable: show entire trajectory section (frames + title + actions).
                       When False, only strategy text is rendered.
    strategy_enable:   show strategy text block above the trajectory.
    """
    if not trajectory_enable:
        return compose_text_image(
            strategy_text or "", font_size=font_size, scale=scale
        )

    imgs = list(trajectory_images)
    if thumbnail_size:
        imgs = [i.resize((thumbnail_size, thumbnail_size), Image.LANCZOS) for i in imgs]
    img_w, img_h = imgs[0].width, imgs[0].height
    n = len(imgs)

    font = _load_font(font_size)

    tmp = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    line_h = tmp_draw.textbbox((0, 0), "Ag", font=font)[3] + 2

    line_w = 2
    label_height = 20
    header_width = 60
    section_title_height = 22 if show_title else 0
    if action_type == "arrow":
        labels = [ACTION_ARROWS.get(a, a) for a in actions] + [""]
    else:
        labels = actions + [""]
    if len(labels) < n:
        labels.extend([""] * (n - len(labels)))

    label_padding = 8
    cell_widths = []
    for image, label in zip(imgs, labels):
        bbox = tmp_draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        cell_widths.append(max(image.width, label_width + 2 * label_padding))
    content_width = sum(cell_widths) + (n - 1) * line_w
    total_width = line_w + header_width + line_w + content_width + line_w
    traj_table_height = line_w + img_h + line_w + label_height + line_w

    if strategy_enable and strategy_text:
        strat_lines = _wrap_text(strategy_text, font, total_width - (line_w + 4) * 2, tmp_draw)
        strat_block_h = len(strat_lines) * line_h + 4
    else:
        strat_lines = []
        strat_block_h = 0

    total_height = strat_block_h + section_title_height + traj_table_height
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 2
    for sl in strat_lines:
        draw.text((line_w + 4, y), sl, fill=(0, 0, 0), font=font)
        y += line_h

    traj_title_y = strat_block_h
    if show_title:
        tb = draw.textbbox((0, 0), "Trajectory example:", font=font)
        draw.text(
            (line_w + 4, traj_title_y + (section_title_height - (tb[3] - tb[1])) // 2),
            "Trajectory example:", fill=(0, 0, 0), font=font,
        )

    ty = traj_title_y + section_title_height
    draw.rectangle(
        [0, ty, total_width - 1, ty + traj_table_height - 1],
        outline=(0, 0, 0),
        width=line_w,
    )
    hx = line_w + header_width
    draw.rectangle([hx, ty, hx + line_w - 1, ty + traj_table_height - 1], fill=(0, 0, 0))
    hy = ty + line_w + img_h
    draw.rectangle([0, hy, total_width - 1, hy + line_w - 1], fill=(0, 0, 0))

    for label, cx, cy, max_w in [
        ("state", line_w, ty + line_w, header_width),
        ("action", line_w, hy + line_w, header_width),
    ]:
        bb = draw.textbbox((0, 0), label, font=font)
        lx = cx + (max_w - (bb[2] - bb[0])) // 2
        h_region = img_h if label == "state" else label_height
        ly = cy + (h_region - (bb[3] - bb[1])) // 2
        draw.text((lx, ly), label, fill=(0, 0, 0), font=font)

    x = line_w + header_width + line_w
    for i, im in enumerate(imgs):
        cell_width = cell_widths[i]
        image_x = x + (cell_width - im.width) // 2
        canvas.paste(im, (image_x, ty + line_w))
        lbl = labels[i]
        bb = draw.textbbox((0, 0), lbl, font=font)
        label_x = x + (cell_width - (bb[2] - bb[0])) // 2
        label_y = hy + line_w + (label_height - (bb[3] - bb[1])) // 2
        draw.text(
            (label_x, label_y), lbl, fill=(0, 0, 0), font=font,
        )
        x += cell_width
        if i < n - 1:
            draw.rectangle([x, ty, x + line_w - 1, ty + traj_table_height - 1], fill=(0, 0, 0))
            x += line_w

    return resize_skill_image(canvas, scale)


def compose_text_image(
    strategy_text: str,
    max_width: int = 600,
    font_size: int = 15,
    padding: int = 10,
    scale: float = 1.0,
) -> Image.Image:
    """Render strategy text as a plain text image (white background, black text)."""
    font = _load_font(font_size)

    tmp = Image.new("RGB", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    text_w = max_width - 2 * padding
    lines = _wrap_text(strategy_text, font, text_w, tmp_draw)
    line_h = tmp_draw.textbbox((0, 0), "Ag", font=font)[3] + 2

    img_h = len(lines) * line_h + 2 * padding
    canvas = Image.new("RGB", (max_width, img_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += line_h

    return resize_skill_image(canvas, scale)
