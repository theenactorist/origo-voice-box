import asyncio
import threading

import pytest

from backend.utils import inference


@pytest.fixture(autouse=True)
def policy(monkeypatch):
    monkeypatch.setattr("backend.utils.platform_detect.is_apple_silicon", lambda: True)
    monkeypatch.setattr(inference, "release_mlx_cache", lambda: None)
    inference._backends.clear()
    yield
    inference._backends.clear()


class Backend:
    def __init__(self):
        self.loaded = False
        self.unloads = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def is_loaded(self):
        return self.loaded

    def unload_model(self):
        self.loaded = False
        self.unloads += 1

    async def load_model(self):
        self.loaded = True

    async def generate(self):
        await self.load_model()
        self.entered.set()
        await self.release.wait()
        assert self.loaded
        return "audio"


@pytest.mark.asyncio
async def test_nested_load_and_concurrent_models_are_serialized():
    first, second = [inference.manage_backend(Backend()) for _ in range(2)]
    a = asyncio.create_task(first.generate())
    await asyncio.wait_for(first.entered.wait(), 1)
    b = asyncio.create_task(second.generate())
    await asyncio.sleep(0)
    assert not second.entered.is_set()
    assert inference.backend_is_busy(first)
    first.release.set()
    assert await a == "audio"
    await asyncio.wait_for(second.entered.wait(), 1)
    assert not first.loaded
    second.release.set()
    assert await b == "audio"
    assert not inference.any_backend_busy()
    assert not second.loaded


@pytest.mark.asyncio
async def test_cancel_does_not_orphan_noncooperative_thread():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class ThreadBackend(Backend):
        async def generate(self):
            await self.load_model()

            def work():
                started.set()
                release.wait(timeout=2)
                assert self.loaded
                finished.set()

            await asyncio.to_thread(work)

    first = inference.manage_backend(ThreadBackend())
    second = inference.manage_backend(Backend())
    a = asyncio.create_task(first.generate())
    assert await asyncio.to_thread(started.wait, 1)
    a.cancel()
    b = asyncio.create_task(second.generate())
    await asyncio.sleep(0)
    assert not a.done()
    assert not second.entered.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await a
    assert finished.is_set()
    await asyncio.wait_for(second.entered.wait(), 1)
    second.release.set()
    await b
    assert not first.loaded


@pytest.mark.asyncio
async def test_loading_another_model_unloads_previous_weights():
    first, second = [inference.manage_backend(Backend()) for _ in range(2)]
    await first.load_model()
    assert first.loaded
    await second.load_model()
    assert second.loaded
    assert not first.loaded


@pytest.mark.asyncio
async def test_error_releases_model_and_lease():
    class Failing(Backend):
        async def generate(self):
            await self.load_model()
            raise ValueError("bad generation")

    backend = inference.manage_backend(Failing())
    with pytest.raises(ValueError, match="bad generation"):
        await backend.generate()
    assert not backend.loaded
    assert not inference.any_backend_busy()


@pytest.mark.asyncio
async def test_transaction_keeps_selected_size_until_generation():
    class Sized(Backend):
        async def load_model(self, size):
            self.size = size
            self.loaded = True

        async def generate(self):
            return self.size

    backend = inference.manage_backend(Sized())

    async def request(size):
        async with inference.model_session(backend):
            await backend.load_model(size)
            await asyncio.sleep(0)
            return await backend.generate()

    assert await asyncio.gather(request("1.7B"), request("0.6B")) == ["1.7B", "0.6B"]
    assert not backend.loaded


@pytest.mark.asyncio
async def test_prompt_error_between_load_and_generate_releases_weights():
    backend = inference.manage_backend(Backend())

    async def prepare():
        async with inference.model_session(backend):
            await backend.load_model()
            raise ValueError("missing profile")

    with pytest.raises(ValueError, match="missing profile"):
        await prepare()
    assert not backend.loaded
    assert not inference.any_backend_busy()
