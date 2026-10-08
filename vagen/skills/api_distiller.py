"""API-based skill distiller using external LLM."""

import base64
import io
import re
from typing import Any, Dict, List, Optional

from .distill_prompt import build_distill_prompt, build_distill_images


def parse_strategy(response: str) -> str:
    match = re.search(r"<strategy>(.*?)</strategy>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _image_to_content(image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class ViSkillDistiller:

    def __init__(
        self,
        model: str = "o3",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            kwargs = {}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def distill(self, case_data: dict) -> Dict[str, Any]:
        messages = self._build_messages(case_data)
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                )
                text = response.choices[0].message.content or ""
                return {"strategy": parse_strategy(text), "raw_response": text}
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {"strategy": "", "raw_response": f"error: {e}"}
        return {"strategy": "", "raw_response": ""}

    def _build_messages(self, case_data: dict) -> List[Dict[str, Any]]:
        actions = case_data.get("trajectory", [])
        task_desc = case_data.get("task_description", "")
        env_type = case_data.get("env_type", "")
        init_frame = case_data.get("init_frame")
        trajectory_image = case_data.get("trajectory_image")

        has_traj_img = trajectory_image is not None
        prompt_text = build_distill_prompt(
            task_desc,
            actions,
            env_type=env_type,
            has_trajectory_image=has_traj_img,
            env_id=case_data.get("env_id", ""),
        )
        images = build_distill_images(init_frame, trajectory_image)

        # Build multimodal content: replace <image> placeholders
        content = []
        parts = prompt_text.split("<image>")
        img_idx = 0
        for i, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if i < len(parts) - 1 and img_idx < len(images):
                content.append(_image_to_content(images[img_idx]))
                img_idx += 1

        return [{"role": "user", "content": content}]
