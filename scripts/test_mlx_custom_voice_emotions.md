# Controlled MLX CustomVoice delivery test

Use an already prepared Python environment with MLX Audio 0.4.1, MLX, NumPy,
Transformers, and SoundFile. The runner installs nothing and accepts only a
complete local **1.7B CustomVoice 4-bit or 8-bit** model snapshot. It forces Hugging Face
and Transformers offline before importing them.

```sh
python scripts/test_mlx_custom_voice_emotions.py \
  --model-path /absolute/path/to/local/model/snapshot \
  --output-dir /absolute/path/to/new/controlled-ryan-tests
```

The destination must be new or empty. The runner writes five float32 WAV files
(neutral, fear, happiness, confusion, neutral repeat), `manifest.json`, and
`runtime.log`. **It never plays audio.** Library console output is redirected to
the log. `--help` and snapshot preflight do not import the MLX runtime.

All cases use Ryan, English, the same sentence (“I did not expect to see you here
today.”), seed 42, and identical sampling settings. Only the separate `instruct`
argument changes. Token generation is capped at 256; MLX Audio may impose its
own tighter text-length cap. When supported, the runner requests a 5 GiB MLX
graph-evaluation memory guideline and a 256 MiB free-cache threshold before
loading. In the inspected MLX 0.31.2 API, `set_memory_limit(limit)` exposes no
`strict` argument and explicitly permits exceeding the guideline. The free
cache threshold triggers reclamation on a subsequent allocation. These settings
are not hard allocation or OS RSS caps and do not cap other applications.
The manifest records the requested settings and whether measured MLX peak
memory exceeded the guideline. MLX needs GPU access;
a restricted sandbox may prevent even imports.

For the inspected MLX Audio 0.4.1 implementation, the effective cap is
`min(max_tokens, max(75, len(tokenizer.encode(text)) * 6))`. The manifest records
this expected cap, actual returned token counts, and flags counts at or above
the cap as potentially truncated. A case marked `complete` means its WAV and
metrics were saved; it does not prove the sentence finished or the emotion worked.

The manifest is saved atomically before loading, before each case, at observed
input-preparation calls, and after each saved file. It contains runtime versions,
model config hash, prompts, settings, load/generation times, MLX memory readings,
audio duration, RMS, peak, full-scale crossing counts, sample/file hashes, and
comparisons against neutral. Float WAV output preserves values outside ±1
without clipping them during writing. Full-scale crossing counts cannot detect
clipping that occurred earlier in the model.

Where source inspection permits, a temporary wrapper observes the **actual**
`_prepare_generation_inputs` call and records its exact arguments, returned
shapes, and a hash of the actual combined input embeddings passed to the talker.
Hashing materializes those embeddings and adds a synchronization point to the
reported generation time; it does not change their values or the RNG state.
The wrapper forwards all arguments unchanged and is removed afterward. Failure
to install the trace is explicitly recorded; successful tracing establishes
instruction receipt, not perceptual success. Emotion labels describe requested
conditions. Waveform differences alone do not prove recognizable emotions.

Check the final `reproducibility.neutral_repeat` result before attributing other
differences to instructions. Equal seed and settings cannot promise identical
results across runtime versions or hardware. **Audio sample differences** in the
neutral repeat limit attribution of other waveform differences to instructions.
Use `float32_samples_identical` and `sample_difference_rms` for this check: a WAV
container hash difference alone does not limit attribution, because metadata can
differ even when audio samples match.
Listening evaluation remains
pending authorization; this runner does not provide playback or previews.

The app-compatible runtime pins MLX 0.31.2, MLX Audio 0.4.1, MLX LM 0.31.1,
Transformers 4.57.3, and Hugging Face Hub 0.36.2. MLX Audio's declared dependency
minimums conflict with these app pins, so its package is installed separately
with `--no-deps` when preparing that environment. Successful execution here
validates only this TTS path, not every feature of those packages. This runner
records the versions actually used and does not install or change packages.
