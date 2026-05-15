# Transcripts

Transcribe long videos (1h–24h+) from YouTube URLs or local files.
Outputs `.txt`, `.srt`, `.vtt`, and `.json` per video.

## What it does

```
YouTube URL ──▶ try creator-uploaded subs (free, instant)
            └─▶ fall back to: yt-dlp audio  ──┐
                                              ├──▶ WhisperX (large-v3, GPU) ──▶ .txt .srt .vtt .json
local file  ──▶ ffmpeg → 16 kHz mono WAV  ────┘
```

- **YouTube fast-path:** if the creator uploaded subtitles, they're downloaded directly — no Whisper run.
- **Whisper path:** `large-v3` via WhisperX gives word-level timestamps and handles multi-hour audio via VAD chunking.
- **Languages:** auto-detects per chunk — works for English, Hindi, and Hindi-English code-switched audio.

## Setup (Windows + NVIDIA GPU) — verified working

```powershell
# 1. ffmpeg (system-wide). Restart shell or refresh $env:Path after install:
#       $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + `
#                   [System.Environment]::GetEnvironmentVariable("Path","User")
winget install Gyan.FFmpeg

# 2. Python venv (use Python 3.12 — 3.13/3.14 wheels for ctranslate2/pyannote lag)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip wheel setuptools

# 3. CUDA torch FIRST. whisperx 3.8 + pyannote-audio 4.0 require torch~=2.8, which lives on cu126/cu128 indexes.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# 4. Then whisperx + yt-dlp
pip install -r requirements.txt

# 5. Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The first `large-v3` run downloads the ~3 GB model into `./models/large-v3/` (not the HF cache — see "Why a local models dir" below).

## Usage

```powershell
# YouTube — tries creator subs first, falls through to Whisper
python transcribe.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Skip the creator-subs check (force Whisper)
python transcribe.py --no-fastpath "https://www.youtube.com/watch?v=VIDEO_ID"

# Local video file
python transcribe.py "C:\path\to\lecture.mp4"

# Force a language (skips auto-detect — slightly faster)
python transcribe.py --language en "C:\path\to\talk.mp4"

# Lower batch size if you OOM on a smaller GPU
python transcribe.py --batch-size 2 "long_video.mp4"

# CPU-only fallback (slow — for testing or no-GPU machines)
python transcribe.py --device cpu --model small "clip.mp4"
```

Outputs land in `./transcripts/<video_id_or_filename>/`. The intermediate `.wav` is kept so you can rerun with a different model without re-downloading.

## VRAM guide

The script defaults to `int8_float16` compute type and **auto-halves `batch_size` on CUDA OOM** (with a model reload between attempts). On a 4 GB GPU, English clips often work at `batch-size 4` but Hindi / mixed-language audio with longer batches can OOM — `batch-size 2` is the safer default.

| VRAM | Recommended flags |
|---|---|
| 4 GB (RTX 3050 Laptop, etc.) | `--batch-size 2 --compute-type int8_float16` (defaults) |
| 6 GB | `--batch-size 4 --compute-type int8_float16` |
| 8 GB | `--batch-size 8 --compute-type int8_float16` |
| 10-12 GB | `--batch-size 16 --compute-type float16` |
| 16 GB+ | `--batch-size 32 --compute-type float16` |
| <4 GB | `--model medium` or `--model large-v3-turbo` |

If you OOM at `batch_size=1`, swap the model:
- `--model large-v3-turbo` -- ~2x faster, lighter decoder, small accuracy hit
- `--model medium` -- smaller model, lower accuracy
- `--device cpu` -- slow but no VRAM limit

## Long videos (12h–24h+)

WhisperX chunks internally via VAD, so a single run on a 24-hour file works if you have the disk space for the intermediate WAV (~1 GB/hr at 16 kHz mono). Rough timing:

| GPU | Throughput (large-v3, int8_float16) | 24 h video ≈ |
|---|---|---|
| RTX 3050 Laptop (4 GB) | ~3–5× realtime | ~6–8 h |
| RTX 3060 (12 GB, float16) | ~10× realtime | ~2.5 h |
| RTX 4090 | ~30× realtime | ~50 min |
| CPU only | ~0.3× realtime | days — don't |

For very long files, split first for restartability:

```powershell
ffmpeg -i full.wav -f segment -segment_time 3600 -c copy chunks/part_%03d.wav
```

Then transcribe each chunk and concatenate the SRTs with timestamp offsets (one chunk per hour).

## Hindi / English mix notes

- Pass `--language` only if you know the whole video is one language. For mixed content, let it auto-detect per chunk.
- Whisper outputs Hindi in **Devanagari** (हिन्दी), not romanised. For Hinglish/romanised, post-process with `indic-transliteration`.
- Code-switched audio is the weakest accuracy point. Spot-check the first 10 minutes before launching a 24h job.
- If Hindi-heavy content is poor, swap the model for **AI4Bharat IndicWhisper** (same WhisperX interface, fine-tuned on Indian languages).

## Cloud escape hatch

When local isn't fast enough or accurate enough:
- **Groq** (`whisper-large-v3`, ~$0.04/audio-hour, very fast)
- **Deepgram Nova-3** (~$0.26/audio-hour, strong on code-switching)
- **AssemblyAI Universal-2** (~$0.37/audio-hour, good diarization)

Not wired into `transcribe.py` yet — add a `--cloud` flag if you start needing it.

## Gotchas / Notes

### Why a local `models/` directory (instead of the HF cache)
HuggingFace's default cache uses symlinks to deduplicate blobs. On Windows without **Developer Mode** (or admin rights), `os.symlink` fails with `WinError 1314: A required privilege is not held by the client`. The script sidesteps this by pre-downloading models to `./models/<name>/` with `snapshot_download(local_dir=...)`, which writes plain files. If you'd rather use the HF cache, enable Developer Mode in Settings → Privacy & Security → For developers.

### `WARNING: No supported JavaScript runtime` from yt-dlp
As of late 2025, YouTube extraction works best when yt-dlp can run JS (e.g., `deno`, `node`). Without it, you get the warning and *some* video formats may be unavailable, but **subtitle download and most audio formats still work**. To silence it, install Deno: `winget install DenoLand.Deno`.

### `torchvision::nms does not exist` after upgrading whisperx
You upgraded `torch` without upgrading `torchvision` to a matching version. They must move together — `torch 2.8.0` ↔ `torchvision 0.23.0`. Reinstall both from the same `--index-url`.

### `torch.cuda.is_available()` is `False` after `pip install -r requirements.txt`
pip's resolver replaced your CUDA torch with the CPU build from PyPI (because some sub-dep tightened the torch version constraint). Re-run step 3 with `--force-reinstall`.

### Outputs missing on a long run
Check `./transcripts/<id>/` for the intermediate `audio.wav` — if it exists, the audio acquisition succeeded. If only `.wav` exists, Whisper crashed mid-run; the most common cause is OOM. Lower `--batch-size` or switch to `--model large-v3-turbo`.
