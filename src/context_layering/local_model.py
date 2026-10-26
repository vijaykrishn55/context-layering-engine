"""Async interface to a local Hugging Face instruction model (no API keys)."""

from __future__ import annotations

import asyncio
from typing import Any


class LocalModelRunner:
    """Run local Transformers inference through an asynchronous interface.

    Model loading and generation are blocking operations, so they execute in worker
    threads rather than blocking the asyncio event loop. Model inference is serialized
    because a shared model instance is not assumed to be concurrency-safe.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 120,
        trust_remote_code: bool = False,
    ) -> None:
        if max_new_tokens < 1 or max_new_tokens > 2048:
            raise ValueError("max_new_tokens must be between 1 and 2048")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def load(self) -> None:
        """Load tokenizer/model once without blocking the event loop."""
        async with self._load_lock:
            if not self.loaded:
                await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local inference dependencies are missing. Install requirements.txt "
                "and ensure PyTorch is installed for your runtime."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one response asynchronously using the configured local model."""
        await self.load()
        async with self._inference_lock:
            return await asyncio.to_thread(self._generate_sync, system_prompt, user_prompt)

    def _generate_sync(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model is not loaded")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
