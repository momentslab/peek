"""Greedy multi-image captioning with the four VLMs used in the paper.

The k selected frames are passed, in temporal order, as one multi-image user
message followed by the prompt. No grid, timestamps or frame delimiters.
"""
from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def model_family(model_id: str) -> str:
    name = model_id.lower()
    if "qwen2.5-vl" in name:
        return "qwen2_5_vl"
    if "qwen3.5" in name:
        return "qwen3_5"
    if "smolvlm" in name:
        return "smolvlm"
    raise ValueError(f"Unsupported captioning model: {model_id}")


def clean_caption(text: str) -> str:
    """Post-processing applied to every generated caption before scoring."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    for token in ("</think>", "<think>"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.strip()
    for prefix in ("Assistant:", "assistant:"):
        if prefix in cleaned:
            cleaned = cleaned.split(prefix, 1)[1].strip()
    return cleaned


class Captioner:
    """bfloat16 + SDPA on CUDA, greedy decoding, checkpoint generation_config otherwise.

    Only ``do_sample`` is overridden; e.g. Qwen2.5-VL keeps its released
    ``repetition_penalty=1.05``.
    """

    def __init__(
        self,
        model_id: str,
        device: str | torch.device = "cuda",
        *,
        image_max_pixels: int | None = None,
        oom_retry_image_max_pixels: int | None = None,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.family = model_family(model_id)
        self.device = torch.device(device)
        self.image_max_pixels = image_max_pixels
        self.oom_retry_image_max_pixels = oom_retry_image_max_pixels

        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.device.type == "cuda":
            kwargs["attn_implementation"] = "sdpa"
        self.processor = AutoProcessor.from_pretrained(model_id)
        if self.family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.to(self.device).eval()
        config = self.model.generation_config
        config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(config, name):
                setattr(config, name, None)

    def _messages(self, images: list[Image.Image], prompt: str, max_pixels: int | None) -> list[dict]:
        content: list[dict[str, Any]] = []
        for image in images:
            item: dict[str, Any] = {"type": "image", "image": image}
            if max_pixels is not None:
                item["max_pixels"] = int(max_pixels)
            content.append(item)
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _inputs(self, images: list[Image.Image], prompt: str, max_pixels: int | None) -> dict:
        messages = self._messages(images, prompt, max_pixels)
        if self.family == "qwen2_5_vl":
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            return self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            )
        template_kwargs: dict[str, Any] = {}
        if self.family == "qwen3_5":
            template_kwargs["enable_thinking"] = False
            # A template that silently ignores enable_thinking makes Qwen3.5
            # spend its 20 tokens thinking (CIDEr ~4). Fail instead.
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            if "<think>\n\n</think>" not in rendered:
                raise RuntimeError("Qwen3.5 chat template did not disable thinking; check transformers version")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        inputs.pop("token_type_ids", None)
        return inputs

    @torch.inference_mode()
    def _generate(self, images: list[Image.Image], prompt: str, max_new_tokens: int, max_pixels: int | None) -> str:
        inputs = self._inputs(images, prompt, max_pixels)
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    inputs[key] = value.to(device=self.device, dtype=self.model.dtype)
                else:
                    inputs[key] = value.to(device=self.device)
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return clean_caption(text)

    def caption(self, images: list[Image.Image], prompt: str, max_new_tokens: int) -> tuple[str, int | None]:
        """Return ``(caption, image_max_pixels actually used)``."""
        if not images:
            raise ValueError("At least one image is required")
        try:
            return self._generate(images, prompt, max_new_tokens, self.image_max_pixels), self.image_max_pixels
        except torch.OutOfMemoryError:
            if self.oom_retry_image_max_pixels is None:
                raise
            torch.cuda.empty_cache()
            retry = self.oom_retry_image_max_pixels
            return self._generate(images, prompt, max_new_tokens, retry), retry
