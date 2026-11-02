#!/usr/bin/env python3
"""Save controlled, offline Ryan delivery tests; never play audio.

Requires an already installed MLX Audio runtime and a complete local snapshot.
See test_mlx_custom_voice_emotions.md for usage and interpretation limits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback


TEXT = "I did not expect to see you here today."
VOICE = "Ryan"
LANGUAGE = "English"
SEED = 42
SETTINGS = {
    "temperature": 0.9,
    "max_tokens": 256,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "verbose": False,
    "stream": False,
}
NEUTRAL = "Speak in a calm, neutral, matter-of-fact tone without strong emotion."
CASES = [
    ("neutral", NEUTRAL),
    ("fear", "Speak with clear fear and apprehension, a trembling voice and urgent breath."),
    ("happiness", "Speak with clear happiness and delight, a warm smile and lively enthusiasm."),
    ("confusion", "Speak with clear confusion and uncertainty, hesitant pauses and questioning intonation."),
    ("neutral_repeat", NEUTRAL),
]
MEMORY_LIMIT = 5 * 1024**3
CACHE_LIMIT = 256 * 1024**2


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_manifest(path, manifest):
    manifest["updated_at"] = utc_now()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_snapshot(path):
    if not path.is_dir():
        raise ValueError("--model-path must be an existing local snapshot directory.")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("tts_model_type") != "custom_voice" or config.get("tts_model_size") != "1b7":
        raise ValueError("This controlled test requires a 1.7B CustomVoice checkpoint.")
    quantization = config.get("quantization") or config.get("quantization_config") or {}
    if quantization.get("bits") not in (4, 8):
        raise ValueError("This runner requires a 4-bit or 8-bit MLX checkpoint.")
    required = ["tokenizer_config.json", "vocab.json", "merges.txt", "speech_tokenizer/config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    for directory in (path, path / "speech_tokenizer"):
        if not any(p.is_file() for p in directory.glob("*.safetensors")):
            missing.append(str(directory.relative_to(path) / "*.safetensors"))
        index_path = directory / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            for filename in sorted(set(index.get("weight_map", {}).values())):
                if not (directory / filename).is_file():
                    missing.append(str((directory / filename).relative_to(path)))
    if missing:
        raise ValueError("Incomplete local snapshot; missing: " + ", ".join(missing))
    return {
        "path": str(path),
        "config_sha256": sha256_file(config_path),
        "model_type": config.get("model_type"),
        "tts_model_type": config["tts_model_type"],
        "tts_model_size": config["tts_model_size"],
        "quantization": quantization,
    }


def memory_function(mx, name):
    function = getattr(mx, name, None)
    if function is None:
        function = getattr(getattr(mx, "metal", None), name, None)
    return function


def memory_snapshot(mx):
    result = {}
    for name in ("get_active_memory", "get_cache_memory", "get_peak_memory"):
        function = memory_function(mx, name)
        if function is not None:
            result[name.removeprefix("get_") + "_bytes"] = int(function())
    if "peak_memory_bytes" in result:
        result["peak_exceeds_requested_memory_guideline"] = result["peak_memory_bytes"] > MEMORY_LIMIT
    return result


def set_memory_limits(mx):
    limits = {}
    for name, value in (("set_memory_limit", MEMORY_LIMIT), ("set_cache_limit", CACHE_LIMIT)):
        function = memory_function(mx, name)
        item = {"requested_bytes": value, "available": function is not None, "applied": False}
        item["semantics"] = (
            "MLX 0.31.2 graph-evaluation memory guideline; may be exceeded. No strict parameter is exposed; not a hard allocation or OS RSS cap."
            if name == "set_memory_limit" else
            "Free-cache reclamation threshold; excess free cache is reclaimed on a subsequent allocation."
        )
        limits[name] = item
        if function is not None:
            # Abort on a setter error rather than silently omit the requested setting.
            previous = function(value)
            item.update(applied=True, previous_bytes=int(previous) if previous is not None else None)
    return limits


@contextmanager
def trace_conditioning(model, current_case, checkpoint, manifest, mx, np):
    """Observe the real method entry, without altering arguments or its result."""
    cls = type(model)
    name = "_prepare_generation_inputs"
    original = getattr(cls, name, None)
    trace = manifest["conditioning_trace"]
    if original is None:
        trace.update(enabled=False, reason="Runtime has no _prepare_generation_inputs method.")
        checkpoint()
        yield
        return
    try:
        signature = inspect.signature(original)
        if "instruct" not in signature.parameters:
            raise ValueError("Method signature does not expose instruct.")
        trace["signature"] = str(signature)
        trace["source_file"] = inspect.getsourcefile(original)
        source, line = inspect.getsourcelines(original)
        trace["source_first_line"] = line
        trace["source_sha256"] = hashlib.sha256("".join(source).encode()).hexdigest()
    except (TypeError, OSError, ValueError) as exc:
        trace.update(enabled=False, reason=str(exc))
        checkpoint()
        yield
        return

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        case = current_case[0]
        entry = {
            "entered_at": utc_now(),
            "text": bound.arguments.get("text"),
            "speaker": bound.arguments.get("speaker"),
            "language": bound.arguments.get("language"),
            "instruct": bound.arguments.get("instruct"),
            "returned": False,
        }
        case["conditioning_calls"].append(entry)
        checkpoint()
        result = original(self, *args, **kwargs)
        entry["returned"] = True
        entry["returned_shapes"] = [list(item.shape) for item in result if hasattr(item, "shape")]
        # Fingerprint the actual combined input embeddings sent to the talker.
        # This adds a synchronization point, but does not alter values or RNG.
        fingerprint_started = time.perf_counter()
        embeddings = result[0]
        materialized = embeddings.astype(mx.float32)
        mx.eval(materialized)
        entry["input_embeddings"] = {
            "shape": list(embeddings.shape),
            "original_dtype": str(embeddings.dtype),
            "hash_encoding": "contiguous little-endian float32",
            "sha256": hashlib.sha256(np.asarray(materialized).astype("<f4").tobytes()).hexdigest(),
        }
        entry["embedding_fingerprint_seconds"] = time.perf_counter() - fingerprint_started
        checkpoint()
        return result

    had_own_method = name in cls.__dict__
    try:
        setattr(cls, name, observed)
    except (TypeError, AttributeError) as exc:
        trace.update(enabled=False, reason=f"Method wrapper unavailable: {exc}")
        checkpoint()
        yield
        return
    trace.update(enabled=True, mechanism="temporary class-method wrapper; arguments/results unchanged")
    checkpoint()
    try:
        yield
    finally:
        if had_own_method:
            setattr(cls, name, original)
        else:
            delattr(cls, name)


def compare_audio(np, neutral, candidate, reference_case):
    same_shape = neutral.shape == candidate.shape
    result = {
        "reference_case": reference_case,
        "same_sample_count": same_shape,
        "float32_samples_identical": bool(same_shape and np.array_equal(neutral, candidate)),
    }
    if same_shape:
        difference = candidate.astype(np.float64) - neutral.astype(np.float64)
        result["maximum_absolute_sample_difference"] = float(np.max(np.abs(difference)))
        result["sample_difference_rms"] = float(np.sqrt(np.mean(difference**2)))
    return result


def run(args):
    model_path = args.model_path.expanduser().resolve()
    snapshot = validate_snapshot(model_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("--output-dir must be empty; existing results will never be overwritten.")
    manifest_path = output_dir / "manifest.json"
    log_path = output_dir / "runtime.log"
    started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "status": "initializing",
        "created_at": utc_now(),
        "model": snapshot,
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "offline": True,
        "playback": False,
        "text": TEXT,
        "voice": VOICE,
        "language": LANGUAGE,
        "seed": SEED,
        "settings": SETTINGS,
        "output_wav_subtype": "FLOAT",
        "conditioning_trace": {
            "enabled": False,
            "expected_path": "generate_custom_voice -> _generate_with_instruct -> _prepare_generation_inputs -> instruction embeddings",
            "interpretation": "A successful trace proves receipt of the exact instruct argument, not perception of the requested emotion.",
            "embedding_fingerprint": "Hashes actual combined talker input embeddings; materialization adds a synchronization point to timings.",
        },
        "limitations": [
            "Emotion names are requested conditions, not automatically verified perceptual labels.",
            "Different samples/hashes alone do not establish recognizable emotion or voice quality.",
            "WAV file hashes can differ because of container metadata; float32 sample equality is the audio repeat check.",
            "MLX 0.31.2 set_memory_limit is a graph-evaluation guideline and can be exceeded; neither it nor MLX peak memory measures or caps total process/system memory.",
            "A fixed seed does not guarantee identical output on all hardware/runtime versions; neutral_repeat tests this run.",
            "No playback occurs. Perceptual evaluation remains pending user authorization.",
        ],
        "cases": [],
    }
    checkpoint = lambda: save_manifest(manifest_path, manifest)
    checkpoint()
    current_case = [None]

    # Set these before third-party imports. Only an existing filesystem path is
    # passed to the loader; no repository ID or download fallback is accepted.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        with log_path.open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            import mlx.core as mx
            import numpy as np
            import soundfile as sf

            manifest["memory_limits"] = set_memory_limits(mx)
            for package in ("mlx", "mlx-audio", "mlx-lm", "transformers", "numpy", "soundfile"):
                try:
                    manifest["runtime"][package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    manifest["runtime"][package] = "unknown"
            checkpoint()
            from mlx_audio.tts.utils import load_model

            manifest["status"] = "loading"
            manifest["load_started_at"] = utc_now()
            checkpoint()
            load_started = time.perf_counter()
            reset_peak = memory_function(mx, "reset_peak_memory")
            if reset_peak is not None:
                reset_peak()
            try:
                model = load_model(model_path)
            finally:
                manifest["load_seconds"] = time.perf_counter() - load_started
            manifest["load_memory"] = memory_snapshot(mx)
            if model.tokenizer is None or model.speech_tokenizer is None:
                raise RuntimeError("Loader did not initialize the text and speech tokenizers.")
            target_token_count = len(model.tokenizer.encode(TEXT))
            expected_effective_token_cap = min(SETTINGS["max_tokens"], max(75, target_token_count * 6))
            manifest["generation_token_cap"] = {
                "text_token_count": target_token_count,
                "requested_max_tokens": SETTINGS["max_tokens"],
                "expected_effective_max_tokens": expected_effective_token_cap,
                "basis": "Inspected MLX Audio 0.4.1 _generate_with_instruct: min(max_tokens, max(75, len(tokenizer.encode(text)) * 6))",
                "runtime_version_matches_inspected_implementation": manifest["runtime"]["mlx-audio"] == "0.4.1",
                "interpretation": "A returned token count at the cap may indicate truncation; it is not a successful-completion signal.",
            }
            manifest["status"] = "generating"
            checkpoint()
            baseline = None
            baseline_hash = None
            baseline_sample_rate = None
            baseline_embedding_hashes = None
            with trace_conditioning(model, current_case, checkpoint, manifest, mx, np):
                for name, instruct in CASES:
                    case = {
                        "name": name, "status": "generating", "started_at": utc_now(),
                        "text": TEXT, "voice": VOICE, "language": LANGUAGE,
                        "instruct": instruct, "seed": SEED, "settings": SETTINGS.copy(),
                        "conditioning_calls": [],
                    }
                    current_case[0] = case
                    manifest["cases"].append(case)
                    checkpoint()
                    clear_cache = memory_function(mx, "clear_cache")
                    if clear_cache is not None:
                        clear_cache()
                    if reset_peak is not None:
                        reset_peak()
                    case["peak_memory_reset_before_case"] = reset_peak is not None
                    random.seed(SEED)
                    np.random.seed(SEED)
                    mx.random.seed(SEED)
                    generation_started = time.perf_counter()
                    chunks, rates, tokens = [], set(), []
                    for result in model.generate_custom_voice(
                        text=TEXT, speaker=VOICE, language=LANGUAGE, instruct=instruct, **SETTINGS
                    ):
                        mx.eval(result.audio)
                        chunks.append(np.asarray(result.audio, dtype=np.float32).copy())
                        rates.add(int(result.sample_rate))
                        tokens.append(int(result.token_count))
                    case["generation_seconds"] = time.perf_counter() - generation_started
                    case["mlx_memory"] = memory_snapshot(mx)
                    case["generated_token_counts"] = tokens
                    case["expected_effective_token_cap"] = expected_effective_token_cap
                    case["token_counts_at_or_above_expected_cap"] = [count >= expected_effective_token_cap for count in tokens]
                    case["potentially_truncated_at_token_cap"] = any(case["token_counts_at_or_above_expected_cap"])
                    if not chunks or len(rates) != 1:
                        raise RuntimeError("Generation returned no audio or inconsistent sample rates.")
                    audio = np.concatenate(chunks)
                    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
                        raise RuntimeError("Generation returned empty, non-mono, or non-finite audio.")
                    sample_rate = rates.pop()
                    if sample_rate <= 0:
                        raise RuntimeError("Generation returned an invalid sample rate.")
                    if baseline_sample_rate is not None and sample_rate != baseline_sample_rate:
                        raise RuntimeError("Sample rate changed between controlled cases.")
                    if manifest["conditioning_trace"]["enabled"]:
                        calls = case["conditioning_calls"]
                        case["instruct_delivery_verified"] = bool(calls) and all(
                            call["instruct"] == instruct and call["text"] == TEXT
                            and call["speaker"] == VOICE and call["returned"] for call in calls
                        )
                        if not case["instruct_delivery_verified"]:
                            raise RuntimeError("Actual input-preparation arguments did not match this case.")
                    else:
                        case["instruct_delivery_verified"] = None
                    amplitude = np.abs(audio.astype(np.float64))
                    case["audio"] = {
                        "sample_rate": sample_rate,
                        "sample_count": int(audio.size),
                        "duration_seconds": audio.size / sample_rate,
                        "rms": float(np.sqrt(np.mean(audio.astype(np.float64)**2))),
                        "absolute_peak": float(amplitude.max()),
                        "samples_at_or_above_full_scale": int(np.count_nonzero(amplitude >= 1.0)),
                        "fraction_at_or_above_full_scale": float(np.mean(amplitude >= 1.0)),
                        "clipping_note": "FLOAT WAV preserves out-of-range values; this counts full-scale crossings, not proven prior clipping.",
                        "float32_samples_sha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
                    }
                    wav_path = output_dir / f"ryan_{name}.wav"
                    sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="FLOAT")
                    case["file"] = {"path": str(wav_path), "sha256": sha256_file(wav_path), "bytes": wav_path.stat().st_size}
                    if baseline is None:
                        baseline, baseline_hash = audio.copy(), case["file"]["sha256"]
                        baseline_sample_rate = sample_rate
                        baseline_embedding_hashes = [call["input_embeddings"]["sha256"] for call in case["conditioning_calls"]]
                    else:
                        case["comparison_to_neutral"] = compare_audio(np, baseline, audio, "neutral")
                        case["comparison_to_neutral"]["wav_files_identical"] = baseline_hash == case["file"]["sha256"]
                        case["comparison_to_neutral"]["input_embedding_hashes_identical"] = (
                            [call["input_embeddings"]["sha256"] for call in case["conditioning_calls"]] == baseline_embedding_hashes
                            if manifest["conditioning_trace"]["enabled"] else None
                        )
                    case.update(status="complete", completed_at=utc_now())
                    case["completion_note"] = "Complete means the WAV and metrics were saved; it does not establish untruncated speech or perceptual success."
                    checkpoint()
                    del chunks, audio
            repeat = manifest["cases"][-1]
            manifest["reproducibility"] = {
                "neutral_repeat": repeat["comparison_to_neutral"],
                "interpretation": "Differences in neutral-repeat AUDIO SAMPLES limit attribution of other waveform differences to instruction changes alone. A WAV container hash difference alone does not; use float32_samples_identical and sample_difference_rms.",
            }
            manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if current_case[0] is not None and current_case[0]["status"] != "complete":
            current_case[0]["status"] = manifest["status"]
        with log_path.open("a") as log:
            traceback.print_exc(file=log)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        checkpoint()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Existing complete MLX 1.7B CustomVoice 4-bit or 8-bit snapshot")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for five WAVs, manifest.json and runtime.log")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Preflight failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
