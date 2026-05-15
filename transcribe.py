"""
Transcribe a video (YouTube URL or local file) to .txt + .srt + .vtt + .json.

Usage:
    python transcribe.py <url-or-path>
    python transcribe.py --model large-v3 --batch-size 16 <input>
    python transcribe.py --no-fastpath <youtube-url>
    python transcribe.py --language hi <input>
    python transcribe.py --device cpu <input>
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Make stdout line-buffered so [acquire]/[whisper]/[align] markers stay in order
# even when Python is launched with stdout captured by another process.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _free_gpu(device: str) -> None:
    gc.collect()
    if device != "cuda":
        return
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent
OUTPUT_ROOT = PROJECT_ROOT / "transcripts"
MODELS_ROOT = PROJECT_ROOT / "models"
URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# Maps the friendly Whisper model names to faster-whisper HF repos.
# We download into MODELS_ROOT/<name>/ to avoid the HF symlink-cache path,
# which fails on Windows without Developer Mode (WinError 1314).
FASTER_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


def ensure_model_local(name: str) -> str:
    """Pre-download the model to MODELS_ROOT/<name> so we never hit the symlink cache path."""
    if Path(name).exists():
        return name  # already a local path
    repo = FASTER_WHISPER_REPOS.get(name)
    if not repo:
        return name  # unknown name — let faster-whisper try the cache path
    target = MODELS_ROOT / name
    if (target / "model.bin").exists():
        return str(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[model] downloading {repo} -> {target}")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=str(target))
    return str(target)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        sys.exit(f"error: required tool not on PATH: {cmd}")


def require_module(name: str) -> None:
    try:
        __import__(name)
    except ImportError:
        sys.exit(f"error: required Python module not installed: {name}")


YT_DLP = [sys.executable, "-m", "yt_dlp"]


def sanitize(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "out"


def make_outdir(stem: str) -> Path:
    d = OUTPUT_ROOT / sanitize(stem)
    d.mkdir(parents=True, exist_ok=True)
    return d


def vtt_or_srt_to_text(path: Path) -> str:
    """Strip cue timing + tags, return plain spoken text with deduped lines."""
    lines: list[str] = []
    last = ""
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or line.startswith("NOTE"):
            continue
        if "-->" in line or line.isdigit():
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\{[^}]+\}", "", line)
        if line and line != last:
            lines.append(line)
            last = line
    return "\n".join(lines)


def try_youtube_fastpath(url: str, outdir: Path) -> bool:
    """Try to fetch creator-uploaded (not auto-generated) subs. Return True on success."""
    print("[fastpath] checking for creator-uploaded YouTube subtitles...")
    cmd = YT_DLP + [
        "--skip-download",
        "--write-subs",
        "--sub-langs", "en,hi,en.*,hi.*",
        "--sub-format", "vtt",
        "--convert-subs", "srt",
        "-o", str(outdir / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[fastpath] yt-dlp failed: {e}")
        return False
    srts = list(outdir.glob("*.srt"))
    if not srts:
        print("[fastpath] no creator subs found, falling through to Whisper")
        return False
    print(f"[fastpath] got {len(srts)} subtitle file(s) — skipping Whisper")
    for srt in srts:
        txt = srt.with_suffix(".txt")
        txt.write_text(vtt_or_srt_to_text(srt), encoding="utf-8")
    return True


def fetch_audio_from_youtube(url: str, outdir: Path) -> Path:
    print("[acquire] downloading audio from YouTube...")
    out_template = str(outdir / "%(id)s.%(ext)s")
    subprocess.run(
        YT_DLP + [
            "-f", "bestaudio",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
            "-o", out_template,
            url,
        ],
        check=True,
    )
    wavs = sorted(outdir.glob("*.wav"))
    if not wavs:
        sys.exit("error: yt-dlp did not produce a .wav file")
    return wavs[-1]


def extract_audio_from_local(path: Path, outdir: Path) -> Path:
    print(f"[acquire] extracting audio from {path.name}...")
    audio = outdir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
        check=True,
    )
    return audio


def transcribe(
    audio_path: Path,
    outdir: Path,
    model_name: str,
    device: str,
    batch_size: int,
    language: str | None,
    compute_type: str | None,
) -> None:
    try:
        import whisperx
    except ImportError:
        sys.exit("error: whisperx not installed. run: pip install whisperx")

    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"
    local_model = ensure_model_local(model_name)

    print("[whisper] loading audio...")
    audio = whisperx.load_audio(str(audio_path))
    _free_gpu(device)

    # Try transcription; on CUDA OOM, halve batch_size and reload the model.
    current_bs = batch_size
    result = None
    while result is None:
        print(f"[whisper] loading {model_name} on {device} ({compute_type}) from {local_model}")
        model = whisperx.load_model(local_model, device=device, compute_type=compute_type, language=language)
        print(f"[whisper] transcribing (batch_size={current_bs})...")
        try:
            result = model.transcribe(audio, batch_size=current_bs)
        except RuntimeError as e:
            del model
            _free_gpu(device)
            if "out of memory" not in str(e).lower():
                raise
            old_bs = current_bs
            current_bs = max(1, current_bs // 2)
            if current_bs == old_bs:
                sys.exit(
                    "error: CUDA out of memory even at batch_size=1. Try one of:\n"
                    "  --model large-v3-turbo   (~2x faster, lighter decoder)\n"
                    "  --model medium           (smaller model, lower accuracy)\n"
                    "  --device cpu             (slow but no VRAM limit)"
                )
            print(f"[whisper] CUDA OOM at batch_size={old_bs}. Reloading model and retrying at batch_size={current_bs}...")
            continue
        else:
            # Free the whisper model before alignment (alignment loads its own model)
            del model
            _free_gpu(device)

    detected = result.get("language", language or "unknown")
    print(f"[whisper] detected language: {detected}")

    try:
        print("[align] loading alignment model for word-level timestamps...")
        align_model, metadata = whisperx.load_align_model(language_code=detected, device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
        del align_model
        _free_gpu(device)
    except Exception as e:
        print(f"[align] alignment skipped ({e}); using segment-level timestamps")

    # whisperx.align returns a new dict without 'language' -- writers need it
    result["language"] = detected

    print(f"[output] writing to {outdir}")
    writer_args = {"highlight_words": False, "max_line_count": None, "max_line_width": None}
    for fmt in ("txt", "srt", "vtt", "json"):
        writer = whisperx.utils.get_writer(fmt, str(outdir))
        writer(result, str(audio_path), writer_args)


def main() -> None:
    p = argparse.ArgumentParser(description="Transcribe a video to text + subtitles.")
    p.add_argument("input", help="YouTube URL or path to a local video/audio file")
    p.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="device (default: cuda)")
    p.add_argument("--batch-size", type=int, default=2, help="WhisperX batch size (default: 2 -- safe-ish for 4 GB VRAM; auto-halves on CUDA OOM. Bump to 8-16 on bigger GPUs.)")
    p.add_argument("--compute-type", default=None, help="ctranslate2 compute type (default: int8_float16 on cuda for low-VRAM safety, int8 on cpu)")
    p.add_argument("--language", default=None, help="force language code (e.g. en, hi); default: auto-detect")
    p.add_argument("--no-fastpath", action="store_true", help="skip YouTube creator-subs check")
    args = p.parse_args()

    require("ffmpeg")

    if is_url(args.input):
        require_module("yt_dlp")
        stem = re.search(r"(?:v=|youtu\.be/|/shorts/)([\w-]{6,})", args.input)
        outdir = make_outdir(stem.group(1) if stem else "yt_video")
        if not args.no_fastpath and try_youtube_fastpath(args.input, outdir):
            print(f"\nDone. Outputs in: {outdir}")
            return
        audio_path = fetch_audio_from_youtube(args.input, outdir)
    else:
        path = Path(args.input).expanduser().resolve()
        if not path.exists():
            sys.exit(f"error: file not found: {path}")
        outdir = make_outdir(path.stem)
        audio_path = extract_audio_from_local(path, outdir)

    transcribe(audio_path, outdir, args.model, args.device, args.batch_size, args.language, args.compute_type)
    print(f"\nDone. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
