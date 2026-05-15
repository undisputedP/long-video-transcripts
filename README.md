# long-video-transcripts

> Transcribe **1-hour to 24-hour-plus** videos from a YouTube URL or a local file, on a modest Windows GPU. Outputs `.txt`, `.srt`, `.vtt`, `.json` per video. Hindi / English / code-switched audio supported out of the box.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://download.pytorch.org/whl/cu126)
[![Tested on Windows 11](https://img.shields.io/badge/tested-Windows%2011%20%2B%20RTX%203050%204GB-lightgrey.svg)](#tested-configuration)
[![WhisperX 3.8](https://img.shields.io/badge/whisperx-3.8-purple.svg)](https://github.com/m-bain/whisperX)

---

## Table of contents

1. [What it does](#what-it-does)
2. [Sample output](#sample-output)
3. [Quick start](#quick-start)
4. [Installation (Windows + NVIDIA GPU)](#installation-windows--nvidia-gpu)
5. [Usage — PowerShell wrapper](#usage--powershell-wrapper)
6. [Usage — direct Python](#usage--direct-python)
7. [How it works](#how-it-works)
8. [Configuration](#configuration)
9. [Performance](#performance)
10. [Long videos (12 h - 24 h)](#long-videos-12-h---24-h)
11. [Hindi / English code-switching](#hindi--english-code-switching)
12. [Troubleshooting](#troubleshooting)
13. [Cloud escape hatch](#cloud-escape-hatch)
14. [Tested configuration](#tested-configuration)
15. [Acknowledgements](#acknowledgements)

---

## What it does

```
YouTube URL ──▶ try creator-uploaded subs (free, instant)
            └─▶ fall back to: yt-dlp audio  ──┐
                                              ├──▶ WhisperX (large-v3, GPU) ──▶ .txt .srt .vtt .json
local file  ──▶ ffmpeg → 16 kHz mono WAV  ────┘
```

- **YouTube fast-path** — if the creator uploaded subtitles, they're downloaded directly and you're done in seconds. No Whisper run, no model load.
- **Whisper path** — `large-v3` via [WhisperX](https://github.com/m-bain/whisperX) gives word-level timestamps and handles multi-hour audio via VAD chunking.
- **Auto language detection** per chunk — works for English, Hindi, and Hindi-English code-switched audio without forcing a `--language` flag.
- **Built for low VRAM** — defaults are tuned for a 4 GB GPU using `int8_float16` compute type. Auto-retries with smaller batches on CUDA OOM.
- **Stall-aware wrapper** — the PowerShell entry point shows audio length, ETA, and prints a heartbeat with live GPU stats every 90 s of stdout silence so you know whether a long job is alive or wedged.

---

## Sample output

A 2-minute slice of a Hindi YouTube video with English code-switching. **No translation** — Hindi is rendered in its native Devanagari script, English words stay in Latin letters, exactly as spoken.

**`audio.srt`** (cropped):
```srt
1
00:00:00,031 --> 00:00:21,485
ही होगा कि ये discovery आकेर हुई कैसे बट इससे भी पहले हमारे सामने कुछ questions है question ये है कि यूरोप से इंडिया जाने का ये एक मात्र रास्ता थोड़ी ना है European Mediterranean Sea को cross करके Land Route के थूबी इंडिया आ सकते थे

2
00:00:29,750 --> 00:00:32,853
तो फिर उन्हें अफरीका के नीचे से घुम के आने की जरूत क्यों पड़ी?

3
00:00:33,333 --> 00:00:34,374
तो चली शुरू करते हैं.
```

**Wrapper output (live)** — header + heartbeat during a long run:
```
============================================================
Source : https://www.youtube.com/watch?v=...
Model  : large-v3   batch_size: 2
Probing duration...
Audio  : 13:18:12   ETA: ~03:19:33  (at ~4.0x realtime; YouTube fast-path may finish in seconds)
GPU    : 0% util, 239/4096 MB used
Watchdog: heartbeat after 90s of stdout silence
============================================================
...
[whisper] transcribing (batch_size=2)...
[watchdog] elapsed 04:13 | silent 90s | pid 19708 | GPU 100% / 2768 MB
[watchdog] elapsed 05:43 | silent 90s | pid 19708 | GPU  96% / 2940 MB
[watchdog] elapsed 07:13 | silent 90s | pid 19708 | GPU  99% / 3206 MB
```

---

## Quick start

After [installation](#installation-windows--nvidia-gpu):

```powershell
# Interactive — prompts for a URL or file path
.\Transcribe.ps1

# Or one-shot
.\Transcribe.ps1 "https://www.youtube.com/watch?v=VIDEO_ID"
.\Transcribe.ps1 "C:\path\to\lecture.mp4"
```

Outputs land in `transcripts\<video_id_or_filename>\` as `.txt` / `.srt` / `.vtt` / `.json`. The intermediate `.wav` is preserved so you can re-run with a different model without re-downloading.

---

## Installation (Windows + NVIDIA GPU)

End-to-end verified on **Windows 11 + RTX 3050 Laptop (4 GB VRAM, driver 566.07)**.

### 1. Install ffmpeg system-wide

```powershell
winget install Gyan.FFmpeg
```

`winget` modifies `Path` but doesn't refresh the current shell. Either restart PowerShell, or run:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
            [System.Environment]::GetEnvironmentVariable("Path","User")
```
(The included `Transcribe.ps1` does this for you on each invocation.)

### 2. Create a Python 3.12 venv

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip wheel setuptools
```

> **Why 3.12 specifically?** Python 3.13/3.14 wheels for `ctranslate2` and `pyannote-audio` lag behind the official releases. 3.12 has full coverage as of mid-2026.

### 3. Install CUDA torch FIRST

This step is critical and easy to get wrong. `whisperx 3.8` and `pyannote-audio 4.0` require `torch~=2.8`. The cu121 wheel index caps at `2.5.1`, so use **cu126**:

```powershell
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 `
    --index-url https://download.pytorch.org/whl/cu126
```

For driver 570+ you can swap `cu126` for `cu128`. The torch / torchvision / torchaudio versions must move together.

### 4. Install WhisperX + yt-dlp

```powershell
pip install -r requirements.txt
```

### 5. Verify the GPU is visible

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If this prints `False`, see [Troubleshooting](#troubleshooting). The first `large-v3` run will download the model (~3 GB) into `./models/large-v3/` (intentionally not the HuggingFace cache — see [Why a local models/ directory](#why-a-local-models-directory)).

### 6. (One-time, on first .ps1 run) allow scripts in your user account

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Usage — PowerShell wrapper

`Transcribe.ps1` hides the venv activation, refreshes `PATH`, sets `PYTHONUNBUFFERED=1` for ordered output, prints an ETA banner, and runs a stall watchdog with live GPU stats.

### Common patterns

```powershell
# Interactive prompt (no args)
.\Transcribe.ps1

# One-shot with URL or path as positional arg
.\Transcribe.ps1 "https://www.youtube.com/watch?v=VIDEO_ID"
.\Transcribe.ps1 "C:\path\to\my-video.mp4"

# Batch mode — keeps prompting until you type 'q'
.\Transcribe.ps1 -Loop
```

### All flags

| Flag | Default | Purpose |
|---|---|---|
| `-Source` (positional) | _interactive prompt_ | YouTube URL or local file path |
| `-Model` | `large-v3` | Whisper model name. See [Model selection](#model-selection) |
| `-BatchSize` | `2` | WhisperX batch size. Auto-halves on CUDA OOM |
| `-Language` | _auto-detect_ | Force a language code (`en`, `hi`, ...). Skip detection |
| `-ComputeType` | `int8_float16` | ctranslate2 compute type. Use `float16` if you have ≥10 GB VRAM |
| `-NoFastpath` | _off_ | Skip the YouTube creator-subs check, force Whisper |
| `-Loop` | _off_ | Keep prompting after each transcription |
| `-StallSeconds` | `90` | Watchdog threshold: heartbeat after this many seconds without stdout |
| `-SpeedFactor` | `4.0` | ETA divisor (audioSec / SpeedFactor). Tune for your GPU |

### Examples

```powershell
# Force Whisper (skip creator-subs check)
.\Transcribe.ps1 -NoFastpath "https://www.youtube.com/watch?v=VIDEO_ID"

# Lighter / faster model when 4 GB is too tight for large-v3
.\Transcribe.ps1 -Model large-v3-turbo "long_video.mp4"

# Override auto-detect (slightly faster startup)
.\Transcribe.ps1 -Language hi "hindi_lecture.mp4"

# Quieter watchdog for very long jobs
.\Transcribe.ps1 -StallSeconds 300 "24h_stream.mp4"

# CPU-only fallback (slow)
.\Transcribe.ps1 -Model small -ComputeType int8 "clip.mp4"
```

---

## Usage — direct Python

Skip the wrapper if you want.

```powershell
.\.venv\Scripts\activate
python transcribe.py [OPTIONS] <url-or-path>
```

CLI flags mirror the wrapper:

```text
--model            Whisper model name (default: large-v3)
--device           cuda | cpu (default: cuda)
--batch-size       int (default: 2)
--compute-type     ctranslate2 compute type (default: int8_float16 on cuda)
--language         force language code, e.g. en, hi
--no-fastpath      skip the YouTube creator-subs check
```

---

## How it works

```
                    ┌─────────────────────────────────────────┐
   Source           │ transcribe.py                           │   Outputs
   ──────           │                                         │   ───────
                    │  ┌──────────────┐                       │
   YouTube URL ───▶ │  │ yt-dlp       │── creator subs found ─┼──▶ .srt + .txt
                    │  │  --write-subs│   (fast-path exit)    │
                    │  └──────┬───────┘                       │
                    │         │ no subs                       │
                    │         ▼                               │
                    │  ┌──────────────┐                       │
                    │  │ yt-dlp       │── 16 kHz mono WAV ──┐ │
                    │  │  audio + ff  │                     │ │
                    │  └──────────────┘                     │ │
                    │                                       │ │
   local file ────▶ │  ┌──────────────┐                     │ │
                    │  │ ffmpeg       │── 16 kHz mono WAV ──┤ │
                    │  └──────────────┘                     │ │
                    │                                       ▼ │
                    │  ┌─────────────────────────────────────┐│
                    │  │ WhisperX                            ││
                    │  │  • pyannote VAD chunking            ││
                    │  │  • faster-whisper large-v3 (GPU)    ││
                    │  │  • OOM auto-retry (halve batch)     ││
                    │  │  • wav2vec2 word-level alignment    ││
                    │  └──────────────┬──────────────────────┘│
                    │                 │                       │
                    │                 ▼                       │
                    │     writers: txt / srt / vtt / json ────┼──▶ ./transcripts/<id>/
                    └─────────────────────────────────────────┘
```

**Why these specific tools:**

- **`yt-dlp`** — the only YouTube downloader still in active maintenance with reliable subtitle handling. Auto-grabs creator-uploaded subs in any language with a single regex flag.
- **`faster-whisper`** (CTranslate2 backend) — ~4x faster than vanilla `openai-whisper` and uses ~half the VRAM at the same model size. Supports `int8_float16` compute, which is what makes `large-v3` fit on 4 GB.
- **`WhisperX`** — wraps `faster-whisper` with two killer features for long videos: (1) `pyannote` VAD chunking that handles 24-hour audio without OOM by feeding silence-bounded segments to Whisper, and (2) word-level timestamp alignment via `wav2vec2`, which produces much cleaner SRT timing than Whisper's coarse segment timestamps.
- **`large-v3`** vs. `large-v3-turbo` — turbo is 6x faster but loses 1-2% accuracy and is weaker on multilingual / code-switched audio. Hindi-English code-switching is already a Whisper weak spot, so default to `large-v3` and only switch to turbo if speed is critical.

---

## Configuration

### Model selection

| Model | Size | VRAM (int8_fp16) | Use when |
|---|---|---|---|
| `tiny` | 75 MB | ~1 GB | Quick smoke tests only |
| `base` | 150 MB | ~1 GB | Smoke tests with slightly better quality |
| `small` | 500 MB | ~1.5 GB | CPU fallback, quick drafts |
| `medium` | 1.5 GB | ~2 GB | Best when `large-v3` OOMs at batch_size=1 |
| `large-v3` | 3 GB | ~2.5 GB + activations | **Default.** Best quality, multilingual |
| `large-v3-turbo` | 1.6 GB | ~1.5 GB + activations | When you need 2x throughput on a small GPU |

### VRAM tuning

The script defaults to `int8_float16` and auto-halves `batch_size` on CUDA OOM (with a model reload between attempts).

| VRAM | Recommended flags |
|---|---|
| 4 GB | `-BatchSize 2 -ComputeType int8_float16` (defaults) |
| 6 GB | `-BatchSize 4 -ComputeType int8_float16` |
| 8 GB | `-BatchSize 8 -ComputeType int8_float16` |
| 10-12 GB | `-BatchSize 16 -ComputeType float16` |
| 16 GB+ | `-BatchSize 32 -ComputeType float16` |
| <4 GB | `-Model medium` or `-Model large-v3-turbo` |

If you OOM at `batch_size=1`, swap the model down (the script will tell you):
- `-Model large-v3-turbo` — ~2x faster, lighter decoder, small accuracy hit
- `-Model medium` — smaller model, lower accuracy
- `-ComputeType int8` — int8 only, no fp16 fallback (slower, less VRAM)

### ETA tuning

The wrapper estimates wall-clock time as `audioSeconds / SpeedFactor`. The default `4.0` is calibrated for an RTX 3050 Laptop with `large-v3` + `int8_float16`. Bump it if you have a faster GPU:

| GPU | Suggested `-SpeedFactor` |
|---|---|
| RTX 3050 Laptop / similar 4 GB | `4` (default) |
| RTX 3060 12 GB | `8-10` |
| RTX 4070 / 4080 | `15-20` |
| RTX 4090 | `25-30` |

The wrapper prints `(N.Nx realtime)` at the end of each run so you can calibrate over time.

---

## Performance

Measured on RTX 3050 Laptop (4 GB) + Hindi audio + `large-v3` + `int8_float16`:

| Audio length | `BatchSize` | Wall time | Realtime ratio |
|---|---|---|---|
| 11 s (English JFK clip) | 4 | 22 s | 0.5x (model load dominates) |
| 2 min (Hindi sample) | 2 | 3:02 (with first-time alignment download) | 0.7x |
| 2 min (Hindi sample) | 4 (after OOM retry) | 1:02 (alignment cached) | 1.9x |
| 13 h 18 m (full Hindi video) | 2 | ~6-7 h projected | ~2x |

For long jobs the steady-state speed is what matters; ignore the short-clip numbers (they're dominated by one-time model loading).

---

## Long videos (12 h - 24 h)

WhisperX chunks internally via VAD, so a single run on a 24-hour file works as long as you have:
- **Disk space**: ~1 GB per audio-hour at 16 kHz mono WAV (so a 24 h video → ~24 GB intermediate file)
- **Patience**: at ~2x realtime on 4 GB, a 24 h video takes ~12 h of wall time

For very long files, consider explicit chunking for restartability:

```powershell
ffmpeg -i full.wav -f segment -segment_time 3600 -c copy chunks/part_%03d.wav
```

Then transcribe each chunk and concatenate the `.srt` files with timestamp offsets (one chunk per hour). This way a crash mid-run doesn't lose the whole job.

---

## Hindi / English code-switching

Whisper's strongest weak spot. Notes from real-world testing:

- **Auto-detection works.** Don't force `--language` on mixed audio — let WhisperX detect per chunk.
- **Hindi outputs in Devanagari** (हिन्दी) by default. English code-switched words come out in Latin (Mediterranean, discovery, etc.). This is faithful to the speech, not a translation.
- **Want romanized Hinglish output?** That requires a post-processing transliteration pass — e.g., the `indic-transliteration` library. Not built in.
- **Pure-Hindi accuracy is OK but not great.** For Hindi-heavy content, consider the [AI4Bharat IndicWhisper](https://huggingface.co/ai4bharat) model — same WhisperX interface, fine-tuned on Indian languages.

---

## Troubleshooting

### `torch.cuda.is_available()` is `False` after install

Pip's resolver replaced your CUDA `torch` with the CPU build from PyPI (commonly happens when a sub-dep tightens the version constraint and pip resolves through PyPI's index instead of the CUDA wheel index).

```powershell
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 `
    --index-url https://download.pytorch.org/whl/cu126
```

### `RuntimeError: operator torchvision::nms does not exist`

`torchvision` and `torch` versions diverged. They must move together. For `torch 2.8.0`, install `torchvision 0.23.0` from the same index. See step 3 of installation.

### `OSError: [WinError 1314] A required privilege is not held by the client`

HuggingFace's default cache uses symlinks for blob deduplication. Windows blocks symlinks without Developer Mode. The script avoids this by pre-downloading `faster-whisper` models to `./models/<name>/` with `snapshot_download(local_dir=...)`. If you want to use the HF cache instead (e.g., to share models across projects), enable Developer Mode in **Settings → Privacy & Security → For developers**.

### `RuntimeError: CUDA out of memory` mid-run

The script auto-retries with halved `batch_size`. If it fails at `batch_size=1`:
- `-Model large-v3-turbo` (lighter decoder)
- `-Model medium` (smaller model)
- `-Device cpu` (slow but no VRAM limit)

### `WARNING: No supported JavaScript runtime` from yt-dlp

As of late 2025, YouTube prefers tools that can run JS for some format extraction. Without it, you get the warning and *some* video formats may be unavailable, but **subtitle download and most audio formats still work**. To silence it:

```powershell
winget install DenoLand.Deno
```

### Outputs missing after a long run

Check `transcripts\<id>\` for `audio.wav` (or `<id>.wav` for YouTube). If only the WAV exists, Whisper crashed mid-run — most often OOM. Lower `-BatchSize` or switch to `-Model large-v3-turbo`.

### Why a local `models/` directory

HuggingFace's default cache uses symlinks. On Windows without Developer Mode, `os.symlink` fails (`WinError 1314`). The script sidesteps this by pre-downloading `faster-whisper` models into `./models/<name>/`, which writes plain files. The wav2vec2 alignment models still go to `~/.cache/huggingface/` (HF handles their fallback to copies gracefully — different code path).

### Run is silent for 90 s — is it stalled?

Probably not. The watchdog prints a heartbeat with GPU stats:
```
[watchdog] elapsed 04:13 | silent 90s | pid 19708 | GPU 100% / 2768 MB
```
- **GPU util > 50%** = actively transcribing, just no per-batch logging
- **GPU util 0% + < 500 MB** = the watchdog adds `[WARN GPU idle, may be truly stalled]`

You can tighten with `-StallSeconds 30` for early jobs, or relax with `-StallSeconds 300` once you trust the run.

---

## Cloud escape hatch

When local isn't fast enough or accurate enough — e.g., you want a 5-hour video done in 10 minutes:

| Service | Pricing | Strengths |
|---|---|---|
| [Groq](https://groq.com/) | ~$0.04/audio-hour | Runs `whisper-large-v3` on LPU, very fast |
| [Deepgram Nova-3](https://deepgram.com/) | ~$0.26/audio-hour | Strong on multilingual / code-switching |
| [AssemblyAI Universal-2](https://www.assemblyai.com/) | ~$0.37/audio-hour | Best speaker diarization |

Not wired into `transcribe.py` yet — add a `--cloud groq|deepgram|assemblyai` flag if you start needing it.

---

## Tested configuration

This pipeline is verified end-to-end on:

- **OS**: Windows 11 Home Single Language (build 26200)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop, 4 GB VRAM, driver 566.07
- **Python**: 3.12.8
- **CUDA wheels**: cu126
- **PyTorch**: 2.8.0+cu126, torchvision 0.23.0+cu126, torchaudio 2.8.0+cu126
- **WhisperX**: 3.8.5
- **faster-whisper**: 1.2.1
- **pyannote-audio**: 4.0.4
- **yt-dlp**: 2026.3.17
- **ffmpeg**: 8.1.1 (Gyan.dev build via winget)

Real workloads tested:
- 11-second English clip (JFK speech) — `large-v3` ✓
- 2-minute Hindi-English code-switched clip — `large-v3` ✓ (auto-detected `hi` 0.99 confidence)
- 13.3-hour Hindi YouTube video — `large-v3` in progress
- TED talk URL with creator subs — fast-path ✓ (finished in seconds, no Whisper run)

Untested but should work — Linux, macOS (Apple Silicon needs `mps` device + a different torch wheel), other NVIDIA cards (faster GPUs only need `-SpeedFactor` recalibration).

---

## Acknowledgements

This project is a thin pipeline on top of excellent open-source tools:

- [**WhisperX**](https://github.com/m-bain/whisperX) by Max Bain — VAD chunking + word-level alignment on top of Whisper
- [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN — CTranslate2-backed Whisper, the actual inference engine
- [**pyannote-audio**](https://github.com/pyannote/pyannote-audio) — VAD and (optionally) speaker diarization
- [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) — YouTube downloader and subtitle extractor
- [**OpenAI Whisper**](https://github.com/openai/whisper) — the underlying speech recognition model
- [**ffmpeg**](https://ffmpeg.org/) — universal audio/video conversion

The model used by default ([Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)) is a CTranslate2 conversion of OpenAI's `whisper-large-v3` weights.
