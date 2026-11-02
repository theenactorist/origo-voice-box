# Local instruction-controlled speech check

Tested on 2026-11-02 using an Apple M2 Pro with 16 GiB RAM. This is a
standalone, silent comparison runner, not an integration into the installed
desktop app's engine selector. Perceptual emotion and naturalness remain
unassessed pending listening.

## Model and controls

- Model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- Revision: `41d3337e8b7f2843a75841595fc14e4b9a7a4b96`
- Downloaded payload: 3,080,141,538 bytes; affine 8-bit quantization, group size 64.
- Voice: built-in English male Ryan; no reference recording supplied.
- Same sentence in every case: “I did not expect to see you here today.”
- Same seed (42), punctuation, speaker and sampling settings; only `instruct` changed.
- A fifth case repeats neutral to check reproducibility.

Qwen's [official feature table](https://github.com/QwenLM/Qwen3-TTS) distinguishes
instruction-capable 1.7B CustomVoice from 0.6B. The
[quantized checkpoint](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit)
is an MLX Community conversion. Confusion is a free-form instruction, not a
guaranteed native emotion category.

## Observed results

| Requested delivery | Audio duration | Generation time | Codec tokens |
| --- | ---: | ---: | ---: |
| Neutral | 2.64 s | 4.58 s | 33 |
| Fear | 3.44 s | 2.40 s | 43 |
| Happiness | 5.44 s | 3.33 s | 68 |
| Confusion | 2.56 s | 1.74 s | 32 |
| Neutral repeat | 2.64 s | 1.73 s | 33 |

Loading took 4.45 seconds; the complete runner took 19.53 seconds. All samples
were mono 24 kHz float WAVs, finite and non-silent, with no full-scale crossings.
No case reached the runtime's effective 75-token limit. Whisper Turbo,
loaded only after the TTS process exited, transcribed the exact same spoken
sentence from all four distinct conditions.

A wrapper observed each real `_prepare_generation_inputs` call. Every call
received the intended instruction, unchanged speech text and Ryan speaker.
Combined input-embedding hashes changed for each emotion instruction. The
neutral repeat had identical embeddings and identical decoded audio samples.
Whole-WAV hashes differed because the PEAK chunk contains a timestamp; the
audio data was identical. The other three waveforms and durations differed.
This supports actual conditioning effects, not a conclusion that listeners
will recognize the named emotions or prefer the voice.

Peak MLX allocation was 5,597,687,309 bytes (5.60 GB / 5.21 GiB). MLX 0.31.2
treats the requested 5 GiB limit as a graph-evaluation guideline and can exceed
it. It is not a hard process-memory limit. The external monitor observed one
critical-pressure reading during startup, then normal readings; its sustained
critical-pressure stop was not triggered. These short samples do not establish
the memory requirements of longer material. The TTS subprocess exited after
the run, releasing its allocations.

Pitch estimates are retained in local evidence, but confidence filtering
accepted few frames for several samples; they are insufficient to establish
overall pitch behavior or emotion. Duration, waveform integrity and exact
neutral reproducibility are the stronger objective observations here.

## Reproduction and remaining work

See [runner instructions](../scripts/test_mlx_custom_voice_emotions.md).
The isolated tested environment used MLX 0.31.2, MLX Audio 0.4.1, MLX LM 0.31.1,
Transformers 4.57.3 and Hugging Face Hub 0.36.2. The package metadata conflicts
with the last two app-compatible pins; only this synthesis path was validated.

Generated WAVs, full manifests, transcriptions and resource logs remain local
under the workspace's `verification/explicit-emotion-ryan-8bit` directory.
They are not part of this source repository. No audio was played during this
test. Listening acceptance and desktop integration remain outstanding.
