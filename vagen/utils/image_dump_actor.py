import os
from typing import Optional

import numpy as np
import ray
from PIL import Image


@ray.remote(num_cpus=1)
class ImageDumpActor:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def dump_images(self, step: int, images, compress_level: Optional[int] = None):
        # images: list[None | PIL.Image | np.ndarray | list[...]]
        image_folder = None
        saved = 0

        for idx, img_data in enumerate(images):
            if img_data is None:
                continue
            img_list = img_data if isinstance(img_data, list) else [img_data]
            img_list = [img for img in img_list if img is not None]
            if not img_list:
                continue
            if image_folder is None:
                image_folder = os.path.join(self.base_dir, f"image_{step}")
                os.makedirs(image_folder, exist_ok=True)
            subfolder = os.path.join(image_folder, f"images_{idx}")
            os.makedirs(subfolder, exist_ok=True)
            for img_idx, img in enumerate(img_list):
                img_path = os.path.join(subfolder, f"{img_idx}.png")
                if hasattr(img, "save"):
                    if compress_level is None:
                        img.save(img_path)
                    else:
                        img.save(img_path, compress_level=compress_level)
                elif isinstance(img, np.ndarray):
                    if compress_level is None:
                        Image.fromarray(img).save(img_path)
                    else:
                        Image.fromarray(img).save(img_path, compress_level=compress_level)
                saved += 1

        return image_folder, saved

    def dump_named_images(
        self, step: int, image_groups: dict, compress_level: Optional[int] = None
    ):
        """Save per-sample named images under a deterministic artifact folder."""
        image_folder = os.path.join(self.base_dir, f"artifact_{step}")
        saved = 0
        for name, images in image_groups.items():
            for index, image in enumerate(images):
                if image is None:
                    continue
                sample_folder = os.path.join(image_folder, f"sample_{index:06d}")
                os.makedirs(sample_folder, exist_ok=True)
                path = os.path.join(sample_folder, f"{name}.png")
                output = image if hasattr(image, "save") else Image.fromarray(image)
                if compress_level is None:
                    output.save(path)
                else:
                    output.save(path, compress_level=compress_level)
                saved += 1
        return image_folder if saved else None, saved
