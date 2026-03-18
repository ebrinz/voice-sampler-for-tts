#!/usr/bin/env python3
"""Split YouTube video dialogue by speaker using pyannote + SAM Audio."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split YouTube dialogue by speaker, removing background music."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model-size", choices=["large", "base"], default="large")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--min-segment", type=float, default=0.3)
    parser.add_argument("--merge-gap", type=float, default=0.5)
    parser.add_argument("--max-chunk", type=float, default=20.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-process all segments even if output exists")
    return parser.parse_args()


def check_auth():
    """Verify HuggingFace authentication is available."""
    import os
    if os.environ.get("HF_TOKEN"):
        return
    result = subprocess.run(
        ["huggingface-cli", "whoami"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error: HuggingFace authentication required.", file=sys.stderr)
        print("Either set HF_TOKEN environment variable or run: huggingface-cli login", file=sys.stderr)
        print("You must also accept model terms at:", file=sys.stderr)
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
        print("  https://huggingface.co/facebook/sam-audio-large", file=sys.stderr)
        sys.exit(1)


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: use the last path component or query
    return re.sub(r'[^\w-]', '_', url)[-20:]


def download_audio(url, output_dir):
    """Download audio from YouTube URL as WAV. Returns (wav_path, video_title)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "raw_audio.wav"

    if wav_path.exists():
        print(f"Audio already downloaded: {wav_path}")
        result = subprocess.run(
            ["yt-dlp", "--print", "%(title)s", "--no-download", url],
            capture_output=True, text=True
        )
        title = result.stdout.strip() or "Unknown"
        return wav_path, title

    # Get title first (separate call — mixing --print with download is fragile)
    title_result = subprocess.run(
        ["yt-dlp", "--print", "%(title)s", "--no-download", url],
        capture_output=True, text=True
    )
    title = title_result.stdout.strip() or "Unknown"

    print("Downloading audio...")
    result = subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ac 1",
            "-o", str(wav_path),
            url,
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        wav_path.unlink(missing_ok=True)
        print("Error: yt-dlp failed to download audio.", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)

    if not wav_path.exists():
        candidates = list(output_dir.glob("raw_audio*"))
        if candidates:
            candidates[0].rename(wav_path)
        else:
            print("Error: download succeeded but WAV file not found.", file=sys.stderr)
            sys.exit(2)

    print(f"Downloaded: {wav_path}")
    return wav_path, title


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def merge_segments(segments, merge_gap=0.5):
    """Merge adjacent segments from the same speaker if gap < merge_gap."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end, speaker in segments[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and (start - prev_end) < merge_gap:
            merged[-1] = (prev_start, end, speaker)
        else:
            merged.append((start, end, speaker))
    return merged


def filter_segments(segments, min_duration=0.3):
    """Remove segments shorter than min_duration."""
    return [(s, e, sp) for s, e, sp in segments if (e - s) >= min_duration]


def format_timestamp(seconds):
    """Format seconds as MmSSs for filenames."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m{s:02d}s"


def compute_chunks(duration, max_chunk=20.0, overlap=2.0):
    """Split a duration into overlapping chunks if it exceeds max_chunk."""
    if duration <= max_chunk:
        return [(0.0, duration)]
    chunks = []
    step = max_chunk - overlap
    start = 0.0
    while start < duration:
        end = min(start + max_chunk, duration)
        chunks.append((start, end))
        if end >= duration:
            break
        start += step
    return chunks


def crossfade_stitch(chunks, overlap_samples):
    """Stitch audio chunks with linear crossfade in overlap regions."""
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]
    for chunk in chunks[1:]:
        if overlap_samples > 0 and overlap_samples <= len(result) and overlap_samples <= len(chunk):
            fade_out = torch.linspace(1.0, 0.0, overlap_samples)
            fade_in = torch.linspace(0.0, 1.0, overlap_samples)
            overlap_region = result[-overlap_samples:] * fade_out + chunk[:overlap_samples] * fade_in
            result = torch.cat([result[:-overlap_samples], overlap_region, chunk[overlap_samples:]])
        else:
            result = torch.cat([result, chunk])
    return result


def diarize(wav_path, merge_gap, min_segment):
    """Run pyannote speaker diarization. Returns list of (start, end, speaker) tuples."""
    from pyannote.audio import Pipeline

    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    print("Running diarization...")
    diarization = pipeline(str(wav_path))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    print(f"Found {len(segments)} raw segments")

    segments = merge_segments(segments, merge_gap=merge_gap)
    segments = filter_segments(segments, min_duration=min_segment)

    print(f"After merge/filter: {len(segments)} segments")
    return segments


def write_manifest(output_dir, video_id, title, segments):
    """Write diarization.json manifest."""
    manifest = {
        "video_id": video_id,
        "title": title,
        "segments": [
            {
                "id": i + 1,
                "start": round(s, 2),
                "end": round(e, 2),
                "speaker": sp,
            }
            for i, (s, e, sp) in enumerate(segments)
        ],
    }
    manifest_path = output_dir / "diarization.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {manifest_path}")
    return manifest


def load_sam_audio(model_size, device):
    """Load SAM Audio model and processor."""
    from sam_audio import SAMAudio, SAMAudioProcessor

    model_name = f"facebook/sam-audio-{model_size}"
    print(f"Loading SAM Audio model: {model_name} (this may take a while)...")
    processor = SAMAudioProcessor.from_pretrained(model_name)
    model = SAMAudio.from_pretrained(model_name).to(device)
    model.requires_grad_(False)

    sr = getattr(processor, "audio_sampling_rate", None)
    if sr is None:
        sr = getattr(processor, "sampling_rate", getattr(processor, "sample_rate", None))
        if sr is None:
            print("Warning: could not determine processor sample rate, defaulting to 24000", file=sys.stderr)
    print(f"Model loaded. Output sample rate: {sr or 24000}Hz")
    return model, processor


def _separate_audio(model, processor, device, audio_tensor, sr):
    """Run SAM Audio separation on a single audio tensor."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    torchaudio.save(tmp_path, audio_tensor.unsqueeze(0), sr)

    try:
        inputs = processor(
            audios=[tmp_path],
            descriptions=["a person speaking"]
        ).to(device)

        with torch.inference_mode():
            result = model.separate(inputs)

        target = result.target
        if target.ndim >= 2:
            target = target[0]
        return target.cpu()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def separate_segment(model, processor, device, full_audio, sr, start, end, max_chunk):
    """Separate voice from background for one segment. Handles chunking."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment_audio = full_audio[start_sample:end_sample]

    duration = end - start
    chunks = compute_chunks(duration, max_chunk=max_chunk, overlap=2.0)

    if len(chunks) == 1:
        return _separate_audio(model, processor, device, segment_audio, sr)

    print(f"    Chunking {duration:.1f}s segment into {len(chunks)} parts...")
    separated_chunks = []
    for chunk_start, chunk_end in chunks:
        cs = int(chunk_start * sr)
        ce = int(chunk_end * sr)
        chunk_audio = segment_audio[cs:ce]
        separated = _separate_audio(model, processor, device, chunk_audio, sr)
        separated_chunks.append(separated)

    overlap_samples = int(2.0 * processor.audio_sampling_rate)
    return crossfade_stitch(separated_chunks, overlap_samples=overlap_samples)


def process_all_segments(model, processor, device, wav_path, segments, output_dir, max_chunk, force):
    """Process all diarized segments through SAM Audio."""
    full_audio, sr = torchaudio.load(str(wav_path))
    full_audio = full_audio[0]  # mono

    out_sr = processor.audio_sampling_rate
    total = len(segments)

    for i, (start, end, speaker) in enumerate(segments):
        ts_start = format_timestamp(start)
        ts_end = format_timestamp(end)
        seg_id = f"segment_{i+1:03d}_{ts_start}-{ts_end}"

        speaker_dir = output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        out_path = speaker_dir / f"{seg_id}.wav"

        if out_path.exists() and not force:
            print(f"  [{i+1}/{total}] Skipping (exists): {out_path.name}")
            continue

        duration = end - start
        print(f"  [{i+1}/{total}] {speaker} | {ts_start}-{ts_end} ({duration:.1f}s)")

        separated = separate_segment(
            model, processor, device, full_audio, sr, start, end, max_chunk
        )
        torchaudio.save(str(out_path), separated.unsqueeze(0), out_sr)

    print(f"\nDone! Output in: {output_dir}")


def main():
    args = parse_args()
    check_auth()
    device = get_device(args.device)

    video_id = extract_video_id(args.url)
    output_dir = args.output_dir / video_id
    print(f"Device: {device}")
    print(f"Model: facebook/sam-audio-{args.model_size}")
    print(f"Output: {output_dir}")

    # Stage 1: Download
    wav_path, title = download_audio(args.url, output_dir)
    print(f"Title: {title}")

    # Stage 2: Diarize (with cache)
    manifest_path = output_dir / "diarization.json"
    if manifest_path.exists() and not args.force:
        print(f"Loading cached diarization: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        segments = [(s["start"], s["end"], s["speaker"]) for s in manifest["segments"]]
    else:
        segments = diarize(wav_path, args.merge_gap, args.min_segment)
        manifest = write_manifest(output_dir, video_id, title, segments)

    # Stage 3: Separate with SAM Audio
    model, processor = load_sam_audio(args.model_size, device)
    process_all_segments(
        model, processor, device, wav_path, segments,
        output_dir, args.max_chunk, args.force
    )

    # Summary
    speakers = set(sp for _, _, sp in segments)
    print(f"\nSummary:")
    print(f"  Video: {title}")
    print(f"  Speakers detected: {len(speakers)}")
    print(f"  Segments processed: {len(segments)}")
    print(f"  Output directory: {output_dir}")
    for sp in sorted(speakers):
        sp_segments = [(s, e) for s, e, speaker in segments if speaker == sp]
        total_dur = sum(e - s for s, e in sp_segments)
        print(f"    {sp}: {len(sp_segments)} segments, {total_dur:.1f}s total")


if __name__ == "__main__":
    main()
