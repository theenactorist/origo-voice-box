"""Model-free tests for pinned routing, instructions, cancellation and cleanup.

Run with: python -m unittest backend.tests.test_mlx_custom_voice_backend -v
No MLX runtime, downloaded weights, network connection or audio playback is used.
"""

import asyncio
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from backend import backends
from backend.backends import mlx_custom_voice_backend as adapter


class FakeMLX:
    float32 = np.float32

    def __init__(self):
        self.clear_cache = Mock()
        self.eval = Mock()
        self.set_memory_limit = Mock()
        self.set_cache_limit = Mock()
        self.reset_peak_memory = Mock()
        self.get_peak_memory = Mock(return_value=1234)
        self.random = SimpleNamespace(seed=Mock())


class FakeModel:
    def __init__(self):
        self.calls = []
        self.before_sampling = None
        self.closed = threading.Event()
        self.token_count = 3
        self.invalid_audio = False
        self.drop_instruction = False
        self.tokenizer = SimpleNamespace(encode=lambda text: list(text.split()))
        self.speech_tokenizer = SimpleNamespace(decoder=SimpleNamespace(
            reset_streaming_state=lambda: None, streaming_step=lambda x: x,
        ))

    def _sample_token(self, logits, **kwargs):
        return logits

    def _prepare_generation_inputs(self, text, speaker, language="auto", instruct=None):
        return (np.full((1, 2, 3), len(instruct or ""), dtype=np.float32),)

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        try:
            self._prepare_generation_inputs(
                text=kwargs["text"], speaker=kwargs["speaker"], language=kwargs["language"],
                instruct=None if self.drop_instruction else kwargs["instruct"],
            )
            if self.before_sampling:
                self.before_sampling()
            self._sample_token(np.zeros(1), generated_tokens=[])
            yield SimpleNamespace(
                audio=np.array([0.1, float("nan") if self.invalid_audio else -0.1], dtype=np.float32),
                sample_rate=24000, token_count=self.token_count,
            )
        finally:
            self.closed.set()


class RegistryTests(unittest.TestCase):
    def test_registry_reports_actual_apple_model_and_trained_instruction_support(self):
        with patch.object(backends, "is_apple_silicon", return_value=True):
            large, small = backends._get_qwen_custom_voice_configs()
        self.assertEqual(large.hf_repo_id, backends.QWEN_CUSTOM_VOICE_MLX_REPO)
        self.assertIn("MLX 8-bit", large.display_name)
        self.assertTrue(large.supports_instruct)
        self.assertFalse(small.supports_instruct)
        with patch.object(backends, "is_apple_silicon", return_value=False):
            large, small = backends._get_qwen_custom_voice_configs()
        self.assertEqual(large.hf_repo_id, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.assertFalse(small.supports_instruct)

    def test_apple_factory_does_not_fall_back_when_generic_detection_reports_pytorch(self):
        with patch.dict(backends._tts_backends, clear=True), \
             patch.object(backends, "is_apple_silicon", return_value=True), \
             patch.object(backends, "get_backend_type", return_value="pytorch"), \
             patch("backend.utils.inference.manage_backend", side_effect=lambda backend: backend):
            backend = backends.get_tts_backend_for_engine("qwen_custom_voice")
            self.assertIsInstance(backend, adapter.MLXQwenCustomVoiceBackend)
            self.assertIsNone(backend._legacy)

    def test_non_apple_factory_retains_legacy_backend(self):
        legacy = object()
        module = ModuleType("backend.backends.qwen_custom_voice_backend")
        module.QwenCustomVoiceBackend = Mock(return_value=legacy)
        with patch.dict(backends._tts_backends, clear=True), \
             patch.dict(sys.modules, {module.__name__: module}), \
             patch.object(backends, "is_apple_silicon", return_value=False), \
             patch("backend.utils.inference.manage_backend", side_effect=lambda backend: backend):
            self.assertIs(backends.get_tts_backend_for_engine("qwen_custom_voice"), legacy)


class SnapshotTests(unittest.TestCase):
    def test_only_complete_exact_revision_and_quantization_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = ModuleType("huggingface_hub")
            hub.constants = SimpleNamespace(HF_HUB_CACHE=directory)
            root = Path(directory) / ("models--" + backends.QWEN_CUSTOM_VOICE_MLX_REPO.replace("/", "--")) / "snapshots"
            snapshot = root / backends.QWEN_CUSTOM_VOICE_MLX_REVISION
            (snapshot / "speech_tokenizer").mkdir(parents=True)
            config = {
                "tts_model_size": "1b7", "tts_model_type": "custom_voice",
                "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
                "talker_config": {"spk_id": {"ryan": 3061}},
            }
            (snapshot / "config.json").write_text(json.dumps(config))
            for filename in ("tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors",
                             "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors"):
                (snapshot / filename).write_text("{}")
            backend = adapter.MLXQwenCustomVoiceBackend()
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertEqual(backend._local_snapshot(), snapshot)
                self.assertTrue(backend._is_model_cached("1.7B"))
                (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "missing-shard.safetensors"}}))
                self.assertFalse(backend._is_model_cached("1.7B"))
                (snapshot / "model.safetensors.index.json").unlink()
                config["quantization"]["bits"] = 4
                (snapshot / "config.json").write_text(json.dumps(config))
                self.assertFalse(backend._is_model_cached("1.7B"))
                snapshot.rename(root / "different-revision")
                self.assertFalse(backend._is_model_cached("1.7B"))


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = adapter.MLXQwenCustomVoiceBackend()
        self.mx = FakeMLX()
        self.model = FakeModel()
        self.backend.model = self.model
        self.backend._mx = self.mx
        self.backend._current_model_size = "1.7B"

    async def test_instructions_reach_real_preparation_with_bounded_decode_and_cleanup(self):
        events = []
        self.backend.set_progress_callback(events.append)
        text = "I did not expect to see you here today."
        instruction = "Speak with clear happiness and delight."
        audio, rate = await self.backend.generate(text, {"preset_voice_id": "Ryan"}, seed=42, instruct=instruction)
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(audio, np.array([0.1, -0.1], dtype=np.float32))
        call = self.model.calls[0]
        self.assertEqual(call["text"], text)
        self.assertEqual(call["speaker"], "Ryan")
        self.assertEqual(call["instruct"], instruction)
        self.assertTrue(call["stream"])
        self.assertEqual(call["streaming_interval"], 1.6)
        self.assertEqual(call["max_tokens"], 256)
        self.mx.random.seed.assert_called_once_with(42)
        trace = self.backend.last_generation_info["conditioning_calls"][0]
        self.assertEqual(trace["instruct"], instruction)
        self.assertEqual(trace["text_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertTrue(trace["returned"])
        self.assertEqual(len(trace["input_embeddings_sha256"]), 64)
        self.assertEqual(self.backend.last_generation_info["status"], "complete")
        self.assertFalse(self.backend.is_loaded())
        self.assertTrue(self.model.closed.is_set())
        self.assertNotIn("_sample_token", vars(self.model))
        self.assertNotIn("_prepare_generation_inputs", vars(self.model))
        self.assertGreater(self.mx.clear_cache.call_count, 0)
        self.assertIn("generating", [event["stage"] for event in events])

    async def test_long_input_is_split_before_inference(self):
        await self.backend.generate("A short sentence. " * 50, {}, instruct="Curious.")
        self.assertGreater(len(self.model.calls), 1)
        self.assertTrue(all(len(call["text"]) <= adapter.MAX_CHUNK_CHARS for call in self.model.calls))
        self.assertTrue(all(call["instruct"] == "Curious." for call in self.model.calls))

    async def test_cancellation_waits_for_native_step_and_releases_model(self):
        entered = threading.Event()
        release = threading.Event()

        def blocking_step():
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("Test failed to release simulated native step")

        self.model.before_sampling = blocking_step
        task = asyncio.create_task(self.backend.generate("Hello.", {}, instruct="Afraid."))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done(), "Cancellation must not orphan the worker")
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done(), "Repeated cancellation must still drain")
            with self.assertRaisesRegex(RuntimeError, "running"):
                self.backend.unload_model()
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.model.closed.is_set())
        self.assertFalse(self.backend._worker_active.is_set())
        self.assertFalse(self.backend.is_loaded())
        self.assertNotIn("_sample_token", vars(self.model))
        self.assertNotIn("_prepare_generation_inputs", vars(self.model))

    async def test_silent_instruction_drop_is_rejected_and_unloaded(self):
        self.model.drop_instruction = True
        with self.assertLogs(adapter.logger, level="ERROR"), self.assertRaisesRegex(RuntimeError, "instruction"):
            await self.backend.generate("Hello.", {}, instruct="Fearful.")
        self.assertFalse(self.backend.is_loaded())
        self.assertEqual(self.backend.last_generation_info["status"], "failed")

    async def test_potential_truncation_is_not_returned_as_success(self):
        self.model.token_count = 75
        with self.assertLogs(adapter.logger, level="ERROR"), self.assertRaisesRegex(RuntimeError, "generation limit"):
            await self.backend.generate("Hello.", {}, instruct="Fearful.")
        self.assertFalse(self.backend.is_loaded())
        self.assertEqual(self.backend.last_generation_info["status"], "failed")

    async def test_load_passes_a_local_path_and_never_a_repo_id(self):
        self.backend.model = None
        self.backend._current_model_size = None
        path = Path("/mock-cache") / backends.QWEN_CUSTOM_VOICE_MLX_REVISION
        load = Mock(return_value=self.model)
        packages = {name: ModuleType(name) for name in ("mlx", "mlx.core", "mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils")}
        packages["mlx"].core = self.mx
        packages["mlx.core"] = self.mx
        packages["mlx_audio.tts.utils"].load_model = load
        with patch.dict(sys.modules, packages), patch.object(self.backend, "_local_snapshot", return_value=path), \
             patch.object(adapter, "force_offline_if_cached", return_value=nullcontext()) as offline:
            await self.backend.load_model_async("1.7B")
        load.assert_called_once_with(path)
        offline.assert_called_once_with(True, "MLX CustomVoice pinned snapshot")
        self.assertIsNone(self.backend._legacy)
        self.assertTrue(self.backend.is_loaded())
        self.backend.unload_model()


if __name__ == "__main__":
    unittest.main()
