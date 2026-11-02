"""Cached, pinned MLX CustomVoice 1.7B inference for Apple Silicon.

The app's inference manager serializes this backend with STT and LLM work.
Cancellation is cooperative inside token sampling, and an async cancellation
does not finish until its native worker has stopped. No model download or
PyTorch fallback is permitted for the selected 1.7B checkpoint.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import logging
from pathlib import Path
import random
import threading
import time
import traceback
from typing import Callable, Optional

import numpy as np

from . import LANGUAGE_CODE_TO_NAME, QWEN_CUSTOM_VOICE_MLX_REPO, QWEN_CUSTOM_VOICE_MLX_REVISION
from ..utils.hf_offline_patch import force_offline_if_cached

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER = "Ryan"
MAX_CHUNK_CHARS = 160
MAX_TOKENS = 256
STREAMING_INTERVAL = 1.6  # 20 codec tokens per decoder step at 12.5 Hz.
MEMORY_GUIDELINE_BYTES = 5 * 1024**3
CACHE_LIMIT_BYTES = 256 * 1024**2


class _GenerationCancelled(Exception):
    """Internal worker signal; converted to asyncio cancellation after drain."""


@dataclass
class _WorkerFailure:
    message: str
    cancelled: bool = False


class MLXQwenCustomVoiceBackend:
    """Keep existing preset profiles while using the selected MLX 1.7B model."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size: Optional[str] = None
        self._legacy = None
        self._mx = None
        self._operation_lock = asyncio.Lock()
        self._cancel = threading.Event()
        self._worker_active = threading.Event()
        self._progress_callback: Optional[Callable[[dict], None]] = None
        self.last_generation_info: Optional[dict] = None
        self.device = "mlx"

    def set_progress_callback(self, callback: Optional[Callable[[dict], None]]) -> None:
        """Set a synchronous callback invoked on the inference worker thread.

        Events contain stage, chunk_index/chunk_count when applicable,
        generated_tokens/token_limit during sampling, and audio_seconds after
        decoder yields. Callers must marshal updates onto their event loop.
        """
        self._progress_callback = callback

    def _progress(self, stage: str, **values) -> None:
        if self._progress_callback is not None:
            try:
                self._progress_callback({"stage": stage, **values})
            except Exception:
                logger.warning("CustomVoice progress callback failed", exc_info=True)

    def cancel_current(self) -> None:
        self._cancel.set()

    def _check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise _GenerationCancelled()

    def is_loaded(self) -> bool:
        return self.model is not None or (self._legacy is not None and self._legacy.is_loaded())

    def _get_model_path(self, model_size: str) -> str:
        if model_size == "1.7B":
            return QWEN_CUSTOM_VOICE_MLX_REPO
        if model_size == "0.6B":
            return "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        raise ValueError(f"Unknown CustomVoice model size: {model_size}")

    def _local_snapshot(self) -> Path:
        from huggingface_hub import constants

        path = (
            Path(constants.HF_HUB_CACHE)
            / ("models--" + QWEN_CUSTOM_VOICE_MLX_REPO.replace("/", "--"))
            / "snapshots" / QWEN_CUSTOM_VOICE_MLX_REVISION
        )
        missing_message = (
            "The selected MLX CustomVoice 1.7B 8-bit snapshot is not completely cached "
            f"at {path}. Restore this model snapshot; no alternative model will be downloaded."
        )
        required = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "speech_tokenizer/config.json")
        if any(not (path / filename).is_file() for filename in required):
            raise FileNotFoundError(missing_message)
        config = json.loads((path / "config.json").read_text())
        quantization = config.get("quantization") or config.get("quantization_config") or {}
        if (config.get("tts_model_type") != "custom_voice"
                or config.get("tts_model_size") != "1b7"
                or quantization.get("bits") != 8
                or quantization.get("group_size") != 64
                or quantization.get("mode") != "affine"
                or config.get("talker_config", {}).get("spk_id", {}).get("ryan") != 3061):
            raise ValueError("Cached CustomVoice config does not match the selected 1.7B 8-bit Ryan model.")
        for directory in (path, path / "speech_tokenizer"):
            if not any(p.is_file() for p in directory.glob("*.safetensors")):
                raise FileNotFoundError(missing_message)
            index_path = directory / "model.safetensors.index.json"
            if index_path.is_file():
                weights = json.loads(index_path.read_text()).get("weight_map", {})
                if any(not (directory / filename).is_file() for filename in set(weights.values())):
                    raise FileNotFoundError(missing_message)
        return path

    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        size = model_size or self.model_size
        self._get_model_path(size)  # Validate before choosing either path.
        if size == "0.6B":
            from .base import is_model_cached

            return is_model_cached(self._get_model_path(size))
        try:
            self._local_snapshot()
            return True
        except (OSError, ValueError):
            return False

    def _legacy_backend(self):
        if self._legacy is None:
            from .qwen_custom_voice_backend import QwenCustomVoiceBackend

            self._legacy = QwenCustomVoiceBackend(model_size="0.6B")
        return self._legacy

    async def _await_worker(self, operation):
        """Never return a cancellation while a native inference thread is alive."""
        self._worker_active.set()

        def work():
            try:
                self._check_cancelled()
                result = operation()
                self._check_cancelled()
                return result
            except _GenerationCancelled:
                return _WorkerFailure("Generation cancelled", cancelled=True)
            except Exception as exc:
                # Do not retain a native-model traceback on the asyncio Future:
                # its frames can keep every model tensor alive after unload.
                logger.error("MLX CustomVoice worker failed:\n%s", traceback.format_exc())
                return _WorkerFailure(str(exc))
            finally:
                try:
                    if self._cancel.is_set():
                        self._unload_sync()
                finally:
                    self._worker_active.clear()

        worker = asyncio.create_task(asyncio.to_thread(work))
        try:
            result = await asyncio.shield(worker)
            if isinstance(result, _WorkerFailure):
                self._unload_sync()
                if result.cancelled:
                    raise asyncio.CancelledError()
                raise RuntimeError(result.message)
            return result
        except asyncio.CancelledError:
            self.cancel_current()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue  # Repeated cancel requests must not release the lease early.
                except Exception:
                    break
            if worker.done() and not worker.cancelled():
                try:
                    worker.result()
                except Exception:
                    pass
            self._unload_sync()
            raise

    async def load_model_async(self, model_size: Optional[str] = None) -> None:
        size = model_size or self.model_size
        self._get_model_path(size)
        async with self._operation_lock:
            self._cancel.clear()
            if self.is_loaded() and self._current_model_size == size:
                return
            self._unload_sync()
            self.model_size = size
            if size == "0.6B":
                # Run the existing async adapter within the drained worker so
                # its non-cooperative PyTorch call cannot outlive cancellation.
                await self._await_worker(lambda: asyncio.run(self._legacy_backend().load_model_async(size)))
                self._current_model_size = size
            else:
                await self._await_worker(self._load_model_sync)

    load_model = load_model_async

    def _load_model_sync(self) -> None:
        path = self._local_snapshot()
        self._check_cancelled()
        self._progress("loading", repository=QWEN_CUSTOM_VOICE_MLX_REPO, revision=QWEN_CUSTOM_VOICE_MLX_REVISION)
        try:
            import mlx.core as mx

            self._mx = mx
            # MLX 0.31.2's memory limit is a guideline, not a hard RAM cap.
            if hasattr(mx, "set_memory_limit"):
                mx.set_memory_limit(MEMORY_GUIDELINE_BYTES)
            if hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(CACHE_LIMIT_BYTES)
            with force_offline_if_cached(True, "MLX CustomVoice pinned snapshot"):
                from mlx_audio.tts.utils import load_model

                self.model = load_model(path)
            self._check_cancelled()
            if self.model.tokenizer is None or self.model.speech_tokenizer is None:
                raise RuntimeError("CustomVoice text or speech tokenizer failed to initialize.")
            if not callable(getattr(self.model, "_sample_token", None)):
                raise RuntimeError("Installed MLX Audio lacks the required cancellable token sampling hook.")
            decoder = self.model.speech_tokenizer.decoder
            if not all(callable(getattr(decoder, name, None)) for name in ("reset_streaming_state", "streaming_step")):
                raise RuntimeError("Installed MLX Audio lacks incremental speech decoding support.")
            self._current_model_size = self.model_size = "1.7B"
        except BaseException:
            self._unload_sync()
            raise

    def _unload_sync(self) -> None:
        self.model = None
        self._current_model_size = None
        if self._legacy is not None:
            self._legacy.unload_model()
        gc.collect()
        if self._mx is not None:
            if hasattr(self._mx, "synchronize"):
                self._mx.synchronize()
            self._mx.clear_cache()

    def unload_model(self) -> None:
        if self._worker_active.is_set():
            self.cancel_current()
            raise RuntimeError("CustomVoice is running; cancel and await its operation before unloading.")
        self._unload_sync()

    async def create_voice_prompt(self, audio_path: str, reference_text: str, use_cache: bool = True):
        return {"voice_type": "preset", "preset_engine": "qwen_custom_voice", "preset_voice_id": DEFAULT_SPEAKER}, False

    async def combine_voice_prompts(self, audio_paths: list[str], reference_texts: list[str]):
        from .base import combine_voice_prompts

        return await combine_voice_prompts(audio_paths, reference_texts)

    @contextmanager
    def _sampling_guard(self, model, chunk_index: int, chunk_count: int, token_limit: int):
        original = model._sample_token
        had_override = "_sample_token" in vars(model)

        def guarded(*args, **kwargs):
            self._check_cancelled()
            if "generated_tokens" in kwargs:
                count = len(kwargs.get("generated_tokens") or ())
                self._progress("generating", chunk_index=chunk_index, chunk_count=chunk_count,
                               generated_tokens=count, token_limit=token_limit)
            result = original(*args, **kwargs)
            self._check_cancelled()
            return result

        model._sample_token = guarded
        try:
            yield
        finally:
            if had_override:
                model._sample_token = original
            else:
                delattr(model, "_sample_token")

    @contextmanager
    def _preparation_trace(self, model, chunk_index, text, speaker, instruct, info):
        original = model._prepare_generation_inputs
        signature = inspect.signature(original)
        had_override = "_prepare_generation_inputs" in vars(model)

        def traced(*args, **kwargs):
            self._check_cancelled()
            values = signature.bind(*args, **kwargs)
            values.apply_defaults()
            actual = values.arguments
            entry = {
                "chunk_index": chunk_index,
                "instruct": actual.get("instruct"),
                "speaker": actual.get("speaker"),
                "text_sha256": hashlib.sha256(str(actual.get("text", "")).encode()).hexdigest(),
                "returned": False,
            }
            info["conditioning_calls"].append(entry)
            if actual.get("instruct") != instruct or actual.get("speaker") != speaker or actual.get("text") != text:
                raise RuntimeError("CustomVoice input preparation changed the requested text, speaker or instruction.")
            result = original(*args, **kwargs)
            embeddings = result[0].astype(self._mx.float32)
            self._mx.eval(embeddings)
            entry.update(
                returned=True,
                input_embeddings_shape=list(embeddings.shape),
                input_embeddings_sha256=hashlib.sha256(np.asarray(embeddings).astype("<f4").tobytes()).hexdigest(),
            )
            self._check_cancelled()
            return result

        model._prepare_generation_inputs = traced
        try:
            yield
        finally:
            if had_override:
                model._prepare_generation_inputs = original
            else:
                delattr(model, "_prepare_generation_inputs")

    def _generate_sync(self, text: str, voice_prompt: dict, language: str, seed: Optional[int], instruct: Optional[str]):
        from ..utils.chunked_tts import split_text_into_chunks, concatenate_audio_chunks

        if self.model is None:
            self._load_model_sync()
        mx = self._mx
        model = self.model
        iterator = result = None
        outputs = []
        sample_rate = None
        started = time.perf_counter()
        speaker = voice_prompt.get("preset_voice_id") or DEFAULT_SPEAKER
        chunks = split_text_into_chunks(text, max_chars=MAX_CHUNK_CHARS)
        info = {
            "repository": QWEN_CUSTOM_VOICE_MLX_REPO, "revision": QWEN_CUSTOM_VOICE_MLX_REVISION,
            "speaker": speaker, "instruct": instruct, "seed": seed,
            "streaming_interval": STREAMING_INTERVAL, "max_tokens": MAX_TOKENS,
            "chunks": [], "conditioning_calls": [], "status": "generating",
            "conditioning_trace_note": "Observed actual input preparation and combined embeddings; receipt is not a perceptual emotion judgment. Hashing adds an evaluation synchronization point.",
        }
        self.last_generation_info = info
        try:
            if not chunks:
                raise ValueError("Enter text to generate speech.")
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                mx.random.seed(seed)
            if hasattr(mx, "reset_peak_memory"):
                mx.reset_peak_memory()
            for index, chunk in enumerate(chunks, start=1):
                self._check_cancelled()
                token_limit = min(MAX_TOKENS, max(75, len(model.tokenizer.encode(chunk)) * 6))
                token_count = 0
                chunk_audio = []
                with self._sampling_guard(model, index, len(chunks), token_limit), self._preparation_trace(
                    model, index, chunk, speaker, instruct, info
                ):
                    try:
                        iterator = model.generate_custom_voice(
                            text=chunk, speaker=speaker,
                            language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
                            instruct=instruct, temperature=0.9, max_tokens=MAX_TOKENS,
                            top_k=50, top_p=1.0, repetition_penalty=1.05, verbose=False,
                            stream=True, streaming_interval=STREAMING_INTERVAL,
                        )
                        for result in iterator:
                            self._check_cancelled()
                            mx.eval(result.audio)
                            audio = np.asarray(result.audio, dtype=np.float32).copy()
                            if audio.ndim != 1 or not np.isfinite(audio).all():
                                raise RuntimeError("CustomVoice returned invalid audio.")
                            rate = int(result.sample_rate)
                            if rate <= 0 or (sample_rate is not None and sample_rate != rate):
                                raise RuntimeError("CustomVoice returned inconsistent sample rates.")
                            sample_rate = rate
                            token_count += int(result.token_count)
                            chunk_audio.append(audio)
                            self._progress("decoding", chunk_index=index, chunk_count=len(chunks),
                                           generated_tokens=token_count, token_limit=token_limit,
                                           audio_seconds=sum(len(part) for part in chunk_audio) / rate)
                    finally:
                        if iterator is not None:
                            iterator.close()
                        iterator = result = None
                if not any(call["chunk_index"] == index and call["returned"] for call in info["conditioning_calls"]):
                    raise RuntimeError("CustomVoice did not call the required instruction preparation path.")
                if not chunk_audio:
                    raise RuntimeError("CustomVoice returned no audio.")
                if token_count >= token_limit:
                    raise RuntimeError("Speech reached its generation limit. Shorten the text or delivery instruction and try again.")
                outputs.append(np.concatenate(chunk_audio))
                info["chunks"].append({"characters": len(chunk), "generated_tokens": token_count, "token_limit": token_limit})
                mx.clear_cache()
            self._check_cancelled()
            info["status"] = "complete"
            self._progress("complete", chunk_index=len(chunks), chunk_count=len(chunks))
            return concatenate_audio_chunks(outputs, sample_rate), sample_rate
        finally:
            if iterator is not None:
                iterator.close()
            iterator = result = model = None
            info["generation_seconds"] = time.perf_counter() - started
            if hasattr(mx, "get_peak_memory"):
                info["mlx_peak_memory_bytes"] = int(mx.get_peak_memory())
                info["peak_exceeds_memory_guideline"] = info["mlx_peak_memory_bytes"] > MEMORY_GUIDELINE_BYTES
            if info["status"] != "complete":
                info["status"] = "cancelled" if self._cancel.is_set() else "failed"
            self._unload_sync()

    async def generate(self, text: str, voice_prompt: dict, language: str = "en",
                       seed: Optional[int] = None, instruct: Optional[str] = None):
        async with self._operation_lock:
            self._cancel.clear()
            self.last_generation_info = None
            if self.model_size == "0.6B":
                try:
                    return await self._await_worker(lambda: asyncio.run(self._legacy_backend().generate(
                        text, voice_prompt, language=language, seed=seed, instruct=None,
                    )))
                finally:
                    self._unload_sync()
            return await self._await_worker(lambda: self._generate_sync(text, voice_prompt, language, seed, instruct))
