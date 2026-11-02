# Origo Voice Box 0.5.1 close-out

The selected MLX engine is integrated into the installed desktop app at
`/Applications/Origo Voice Box.app`. Desktop and backend both report 0.5.1;
bundle identifier remains `sh.voicebox.app`. Existing data was backed up before
replacement. A new **Ryan — expressive** preset profile is available.

The 1.7B CustomVoice entry on Apple Silicon resolves only to
`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`, revision
`41d3337e8b7f2843a75841595fc14e4b9a7a4b96`. It reuses the existing model cache,
with no CPU float32 fallback or replacement-model download. Neutral, Fear,
Happiness and Confusion buttons set an editable delivery instruction. Unsupported
models do not expose or submit these controls.

Speech, transcription and text inference share a model lease. The selected
speech adapter decodes in short streaming blocks, checks cancellation during
sampling, drains its worker before releasing the lease, then unloads and clears
MLX allocations. Model loading through generation is one transaction.

## Completed verification

- 18 focused backend tests passed, including cancellation, model-size isolation,
  worker draining, input conditioning and failure cleanup.
- App and web TypeScript checks passed; focused frontend checks passed.
- Native release build completed. Installed bundle passed deep strict signature
  verification, and the previous native signing entitlements were preserved.
- Actual installed-app generation produced a finite, non-silent 24 kHz WAV and
  saved it in history. The observed preparation call received Ryan and the exact
  requested fear instruction. The recording is 3.44 seconds long.
- Cancellation completed in approximately 0.89 seconds after the request,
  recorded a cancelled history entry and left no partial WAV or loaded model.
- Preserved Whisper Turbo transcribed the exact generated sentence. Preserved
  Qwen3 4B completed the text-editing check. Both unloaded after use.
- No test audio was played during integration verification.

The successful speech run took approximately 10.81 seconds including loading;
measured generation was 6.71 seconds and peak MLX allocation was 4.58 GB.
The first attempt was stopped by the external memory-pressure guard; its model
was released, with the backend returning to approximately 339 MB physical
footprint. The retry completed after memory pressure returned to normal. These
short tests do not establish safety for all workloads or long material.

The app is locally ad-hoc signed, not notarized. Emotional labels remain requests,
not guarantees of every performance. No further comparisons or feature expansion
were performed after the user's close-out instruction. Local verification JSON,
WAVs, data backups and packaging audit reports remain outside Git.

Backend packaging used `scripts/repack-origo-backend.py`, preserving the installed
third-party/native payloads while recompiling backend code. The installed sidecar
SHA-256 is `ed00eb2fc9f22703d28c7d4f98c33721fe42d3f05491dec51b6469cdfa9f87c3`.
