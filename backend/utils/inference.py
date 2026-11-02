"""Serialize local model work and release idle MLX allocations.

Cancellation must not release the lease while an inference thread is alive.
Backends with cooperative cancellation stop promptly; other backends finish
their current operation before the next model may enter.
"""

import asyncio
import gc
import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from weakref import WeakKeyDictionary, WeakSet

logger = logging.getLogger(__name__)
_backends = WeakSet()
_locks = WeakKeyDictionary()
_lease = ContextVar("voicebox_inference_lease", default=None)
generation_progress_callback = ContextVar("voicebox_generation_progress_callback", default=None)


def release_mlx_cache():
    """Drop Python references before clearing MLX's allocator cache."""
    gc.collect()
    try:
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()
    except ImportError:
        pass


def backend_is_busy(backend):
    return bool(getattr(backend, "_inference_busy", False))


def any_backend_busy():
    return any(backend_is_busy(backend) for backend in list(_backends))


async def _drain(task):
    """Wait through repeated cancel requests without orphaning model work."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        # Retrieve a failed task's exception even when outer cancellation wins.
        task.exception()


@asynccontextmanager
async def model_session(backend, *, unload_after=True):
    """Hold the selected model across a complete load/prompt/generate transaction."""
    if not getattr(backend, "_inference_managed", False):
        yield
        return
    inherited = _lease.get()
    current = asyncio.current_task()
    if inherited is not None and current in inherited["tasks"]:
        if inherited["backend"] is not backend:
            raise RuntimeError("A model operation cannot start another model before releasing its memory")
        yield
        return
    loop = asyncio.get_running_loop()
    lock = _locks.setdefault(loop, asyncio.Lock())
    async with lock:
        for other in list(_backends):
            if other is not backend and other.is_loaded():
                other.unload_model()
        release_mlx_cache()
        backend._inference_busy = True
        set_progress = getattr(backend, "set_progress_callback", None)
        if callable(set_progress):
            set_progress(generation_progress_callback.get())
        token = _lease.set({"backend": backend, "tasks": {current}})
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            try:
                if unload_after or not succeeded:
                    backend.unload_model()
                    release_mlx_cache()
            finally:
                backend._inference_busy = False
                if callable(set_progress):
                    set_progress(None)
                _lease.reset(token)


async def _invoke(backend, method, args, kwargs):
    lease = _lease.get()
    task = asyncio.create_task(method(*args, **kwargs))
    lease["tasks"].add(task)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancel = getattr(backend, "cancel_current", None)
        if not task.done():
            if callable(cancel):
                cancel()
                task.cancel()
            await _drain(task)
        raise
    finally:
        lease["tasks"].discard(task)


def _managed_method(backend, method, name):
    @wraps(method)
    async def managed(*args, **kwargs):
        async with model_session(backend, unload_after=name in ("generate", "transcribe")):
            return await _invoke(backend, method, args, kwargs)

    return managed


def manage_backend(backend):
    """Apply the memory policy once to each backend on Apple Silicon."""
    from .platform_detect import is_apple_silicon

    if not is_apple_silicon() or getattr(backend, "_inference_managed", False):
        return backend
    backend._inference_managed = True
    backend._inference_busy = False
    _backends.add(backend)
    for name in (
        "load_model", "load_model_async", "generate", "transcribe",
        "create_voice_prompt", "combine_voice_prompts",
    ):
        method = getattr(backend, name, None)
        if method is not None and asyncio.iscoroutinefunction(method):
            setattr(backend, name, _managed_method(backend, method, name))
    return backend
