"""
Local GGUF inference via llama-cpp-python (Path A: llama.cpp / Q4_0, etc.).

Loads lazily on first use. Intended for Gemma-class models; chat template can be
set via AI_LLM_CHAT_FORMAT (e.g. gemma, gemma-2) or left unset for model default.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_llama_lock = threading.Lock()
_llama_instance: Any = None


def _gguf_path_resolved() -> Optional[Path]:
    raw = getattr(settings, "llm_gguf_path", None)
    if not raw:
        return None
    p = Path(raw).expanduser().resolve()
    return p if p.is_file() else None


def local_gguf_configured() -> bool:
    return _gguf_path_resolved() is not None


def _build_llama_kwargs() -> Dict[str, Any]:
    path = _gguf_path_resolved()
    if not path:
        raise FileNotFoundError("GGUF path not configured or file missing")

    n_ctx = int(getattr(settings, "llm_ctx_size", 8192) or 8192)
    n_gpu_layers = int(getattr(settings, "llm_gpu_layers", 0) or 0)
    n_threads = getattr(settings, "llm_cpu_threads", None)
    if n_threads is None:
        import os

        n_threads = max(1, (os.cpu_count() or 4) - 1)
    else:
        n_threads = int(n_threads)

    kwargs: Dict[str, Any] = {
        "model_path": str(path),
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads,
        "verbose": False,
    }
    chat_fmt = getattr(settings, "llm_chat_format", None)
    if chat_fmt and str(chat_fmt).strip():
        kwargs["chat_format"] = str(chat_fmt).strip()
    return kwargs


def _get_llama() -> Any:
    global _llama_instance
    with _llama_lock:
        if _llama_instance is not None:
            return _llama_instance
        try:
            from llama_cpp import Llama
        except ImportError as e:
            logger.warning(
                "llama-cpp-python not installed; local GGUF disabled. pip install llama-cpp-python"
            )
            raise RuntimeError("llama-cpp-python not installed") from e

        kwargs = _build_llama_kwargs()
        logger.info(
            "Loading local GGUF: %s (n_ctx=%s, n_gpu_layers=%s)",
            kwargs["model_path"],
            kwargs["n_ctx"],
            kwargs["n_gpu_layers"],
        )
        _llama_instance = Llama(**kwargs)
        return _llama_instance


def generate_sync(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Blocking chat completion. Run from a worker thread (e.g. asyncio.to_thread).
    """
    if not local_gguf_configured():
        return None

    max_tokens = int(getattr(settings, "llm_max_tokens", 2048) or 2048)
    temperature = float(getattr(settings, "llm_temperature", 0.1) or 0.1)

    try:
        llm = _get_llama()
    except Exception as e:
        logger.error("Failed to load local GGUF: %s", e)
        return None

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        out = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        logger.error("Local GGUF inference error: %s", e)
        return None

    choices = out.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return content if isinstance(content, str) else None


async def generate(system_prompt: str, user_prompt: str) -> Optional[str]:
    return await asyncio.to_thread(generate_sync, system_prompt, user_prompt)
