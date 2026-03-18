# YouTube Speaker Splitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Python script that downloads YouTube audio, diarizes speakers with pyannote, and cleanly separates each speaker's voice from background music using SAM Audio.

**Architecture:** Three-stage pipeline (download → diarize → separate) in one file. Pure helper functions for segment merging, chunking, and crossfading are unit-tested. Pipeline stages are integration-tested manually.

**Tech Stack:** Python 3.12, yt-dlp, torch, torchaudio, sam_audio, pyannote.audio, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-youtube-speaker-splitter-design.md`

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `split_dialogue.py` (skeleton with imports and arg parsing only)
- Create: `tests/test_helpers.py`

- [ ] **Step 1: Create requirements.txt**

```txt
torch
torchaudio
sam_audio
pyannote.audio
yt-dlp
pytest
```

- [ ] **Step 2: Create split_dialogue.py with CLI skeleton**

Write the argparse setup and `main()` entry point. No pipeline logic yet — just parse args and print them.

```python
#!/usr/bin/env python3
"""Split YouTube video dialogue by speaker using pyannote + SAM Audio."""

import argparse
import json
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


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Model: facebook/sam-audio-{args.model_size}")
    print(f"URL: {args.url}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create empty test file**

```python
# tests/test_helpers.py
"""Tests for pure helper functions in split_dialogue.py."""
```

- [ ] **Step 4: Verify skeleton runs**

Run: `python split_dialogue.py "https://www.youtube.com/watch?v=test" --model-size base`
Expected: prints device, model, and URL without errors.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt split_dialogue.py tests/test_helpers.py
git commit -m "feat: project skeleton with CLI arg parsing"
```

---

### Task 2: Auth Check

**Files:**
- Modify: `split_dialogue.py`

- [ ] **Step 1: Add check_auth function**

This runs before any heavy work. Checks that `HF_TOKEN` is set or `huggingface-cli whoami` succeeds.

```python
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
```

- [ ] **Step 2: Call check_auth() at start of main()**

Add `check_auth()` as the first line inside `main()`, before device setup.

- [ ] **Step 3: Verify it works**

Run without auth: should print error and exit.
Run with auth: should pass through to the existing print statements.

- [ ] **Step 4: Commit**

```bash
git add split_dialogue.py
git commit -m "feat: add HuggingFace auth check"
```

---

### Task 3: Stage 1 — YouTube Download

**Files:**
- Modify: `split_dialogue.py`

- [ ] **Step 1: Add extract_video_id helper**

```python
import re

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
```

- [ ] **Step 2: Add download_audio function**

```python
def download_audio(url, output_dir):
    """Download audio from YouTube URL as WAV. Returns (wav_path, video_title)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "raw_audio.wav"

    if wav_path.exists():
        print(f"Audio already downloaded: {wav_path}")
        # Get title separately
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
            "-x",                          # extract audio
            "--audio-format", "wav",        # convert to WAV
            "--postprocessor-args", "ffmpeg:-ac 1",  # mono
            "-o", str(wav_path),
            url,
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        # Clean up partial download
        wav_path.unlink(missing_ok=True)
        print("Error: yt-dlp failed to download audio.", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(2)

    if not wav_path.exists():
        # yt-dlp sometimes appends extensions — find the actual file
        candidates = list(output_dir.glob("raw_audio*"))
        if candidates:
            candidates[0].rename(wav_path)
        else:
            print("Error: download succeeded but WAV file not found.", file=sys.stderr)
            sys.exit(2)

    print(f"Downloaded: {wav_path}")
    return wav_path, title
```

- [ ] **Step 3: Wire into main()**

```python
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
```

- [ ] **Step 4: Test with the actual YouTube URL**

Run: `python split_dialogue.py "https://www.youtube.com/watch?v=II1x9ptMZag"`
Expected: downloads audio to `output/II1x9ptMZag/raw_audio.wav`, prints title.

- [ ] **Step 5: Commit**

```bash
git add split_dialogue.py
git commit -m "feat: stage 1 — download YouTube audio via yt-dlp"
```

---

### Task 4: Segment Merging & Filtering Helpers (TDD)

**Files:**
- Modify: `split_dialogue.py`
- Modify: `tests/test_helpers.py`

These are pure functions — ideal for unit testing.

- [ ] **Step 1: Write failing tests for merge_segments**

```python
# tests/test_helpers.py
from split_dialogue import merge_segments, filter_segments, format_timestamp

def test_merge_adjacent_same_speaker():
    segments = [
        (0.0, 5.0, "speaker_0"),
        (5.3, 10.0, "speaker_0"),  # gap=0.3 < 0.5 -> merge
    ]
    result = merge_segments(segments, merge_gap=0.5)
    assert result == [(0.0, 10.0, "speaker_0")]

def test_no_merge_different_speakers():
    segments = [
        (0.0, 5.0, "speaker_0"),
        (5.3, 10.0, "speaker_1"),
    ]
    result = merge_segments(segments, merge_gap=0.5)
    assert result == [(0.0, 5.0, "speaker_0"), (5.3, 10.0, "speaker_1")]

def test_no_merge_large_gap():
    segments = [
        (0.0, 5.0, "speaker_0"),
        (6.0, 10.0, "speaker_0"),  # gap=1.0 > 0.5 -> no merge
    ]
    result = merge_segments(segments, merge_gap=0.5)
    assert result == [(0.0, 5.0, "speaker_0"), (6.0, 10.0, "speaker_0")]

def test_filter_short_segments():
    segments = [
        (0.0, 5.0, "speaker_0"),
        (6.0, 6.2, "speaker_1"),   # 0.2s < 0.3 -> discard
        (7.0, 12.0, "speaker_0"),
    ]
    result = filter_segments(segments, min_duration=0.3)
    assert result == [(0.0, 5.0, "speaker_0"), (7.0, 12.0, "speaker_0")]

def test_format_timestamp():
    assert format_timestamp(5.2) == "0m05s"
    assert format_timestamp(65.0) == "1m05s"
    assert format_timestamp(0.0) == "0m00s"
    assert format_timestamp(3661.5) == "61m01s"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_helpers.py -v`
Expected: ImportError — functions don't exist yet.

- [ ] **Step 3: Implement merge_segments, filter_segments, format_timestamp**

Add to `split_dialogue.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_helpers.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add split_dialogue.py tests/test_helpers.py
git commit -m "feat: segment merge/filter/format helpers with tests"
```

---

### Task 5: Chunking & Crossfade Helpers (TDD)

**Files:**
- Modify: `split_dialogue.py`
- Modify: `tests/test_helpers.py`

- [ ] **Step 1: Write failing tests for compute_chunks**

```python
# append to tests/test_helpers.py
from split_dialogue import compute_chunks

def test_short_segment_no_chunking():
    """Segment under max_chunk returns single chunk, no overlap."""
    chunks = compute_chunks(duration=15.0, max_chunk=20.0, overlap=2.0)
    assert chunks == [(0.0, 15.0)]

def test_exact_max_chunk_no_split():
    chunks = compute_chunks(duration=20.0, max_chunk=20.0, overlap=2.0)
    assert chunks == [(0.0, 20.0)]

def test_long_segment_chunked():
    """35s segment with 20s max and 2s overlap produces overlapping chunks."""
    chunks = compute_chunks(duration=35.0, max_chunk=20.0, overlap=2.0)
    # Chunk 1: 0-20, Chunk 2: 18-35 (starts at 20-2=18)
    assert len(chunks) >= 2
    # First chunk starts at 0
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 20.0
    # Last chunk ends at duration
    assert chunks[-1][1] == 35.0
    # Overlap between consecutive chunks
    for i in range(1, len(chunks)):
        overlap = chunks[i-1][1] - chunks[i][0]
        assert overlap >= 1.9  # approximately 2s overlap
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_helpers.py::test_short_segment_no_chunking -v`
Expected: ImportError.

- [ ] **Step 3: Implement compute_chunks**

```python
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
```

- [ ] **Step 4: Write failing test for crossfade_stitch**

```python
# append to tests/test_helpers.py
import torch
from split_dialogue import crossfade_stitch

def test_crossfade_single_chunk():
    """Single chunk returns as-is."""
    audio = torch.randn(48000)  # 2s at 24kHz
    result = crossfade_stitch([audio], overlap_samples=0)
    assert torch.equal(result, audio)

def test_crossfade_two_chunks():
    """Two chunks with overlap produce correct length."""
    sr = 24000
    overlap_sec = 2.0
    overlap_samples = int(sr * overlap_sec)
    chunk1 = torch.ones(sr * 20)   # 20s
    chunk2 = torch.ones(sr * 15)   # 15s
    # Expected: 20 + 15 - 2 = 33s
    result = crossfade_stitch([chunk1, chunk2], overlap_samples=overlap_samples)
    expected_len = len(chunk1) + len(chunk2) - overlap_samples
    assert len(result) == expected_len
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_helpers.py::test_crossfade_single_chunk -v`
Expected: ImportError.

- [ ] **Step 6: Implement crossfade_stitch**

```python
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
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_helpers.py -v`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add split_dialogue.py tests/test_helpers.py
git commit -m "feat: chunking and crossfade helpers with tests"
```

---

### Task 6: Stage 2 — Speaker Diarization

**Files:**
- Modify: `split_dialogue.py`

- [ ] **Step 1: Add diarize function**

```python
def diarize(wav_path, merge_gap, min_segment):
    """Run pyannote speaker diarization. Returns list of (start, end, speaker) tuples."""
    from pyannote.audio import Pipeline

    print("Loading diarization model...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    print("Running diarization...")
    diarization = pipeline(str(wav_path))

    # Extract segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    print(f"Found {len(segments)} raw segments")

    # Merge and filter
    segments = merge_segments(segments, merge_gap=merge_gap)
    segments = filter_segments(segments, min_duration=min_segment)

    print(f"After merge/filter: {len(segments)} segments")
    return segments
```

- [ ] **Step 2: Add write_manifest function**

```python
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
```

- [ ] **Step 3: Wire into main() with diarization caching**

After the download stage in `main()`:

```python
    # Stage 2: Diarize (with cache)
    manifest_path = output_dir / "diarization.json"
    if manifest_path.exists() and not args.force:
        print(f"Loading cached diarization: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        segments = [(s["start"], s["end"], s["speaker"]) for s in manifest["segments"]]
    else:
        segments = diarize(wav_path, args.merge_gap, args.min_segment)
        manifest = write_manifest(output_dir, video_id, title, segments)
```

- [ ] **Step 4: Test with the real audio**

Run: `python split_dialogue.py "https://www.youtube.com/watch?v=II1x9ptMZag"`
Expected: downloads (or skips if cached), runs diarization, prints segment count, writes `diarization.json`. Script ends after diarization since Stage 3 isn't wired yet. Run again — should print "Loading cached diarization" and skip pyannote.

- [ ] **Step 5: Commit**

```bash
git add split_dialogue.py
git commit -m "feat: stage 2 — pyannote speaker diarization with merge/filter"
```

---

### Task 7: Stage 3 — SAM Audio Separation

**Files:**
- Modify: `split_dialogue.py`

- [ ] **Step 1: Add load_sam_audio function**

```python
def load_sam_audio(model_size, device):
    """Load SAM Audio model and processor."""
    from sam_audio import SAMAudio, SAMAudioProcessor

    model_name = f"facebook/sam-audio-{model_size}"
    print(f"Loading SAM Audio model: {model_name} (this may take a while)...")
    processor = SAMAudioProcessor.from_pretrained(model_name)
    model = SAMAudio.from_pretrained(model_name).to(device)
    model.requires_grad_(False)

    # Verify processor exposes expected sample rate attribute
    sr = getattr(processor, "audio_sampling_rate", None)
    if sr is None:
        # Fallback: check common attribute names
        sr = getattr(processor, "sampling_rate", getattr(processor, "sample_rate", None))
        if sr is None:
            print("Warning: could not determine processor sample rate, defaulting to 24000", file=sys.stderr)
    print(f"Model loaded. Output sample rate: {sr or 24000}Hz")
    return model, processor
```

- [ ] **Step 2: Add separate_segment function**

This handles a single segment, including chunking for long segments.

```python
def separate_segment(model, processor, device, full_audio, sr, start, end, max_chunk):
    """Separate voice from background for one segment. Handles chunking."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    segment_audio = full_audio[start_sample:end_sample]

    duration = end - start
    chunks = compute_chunks(duration, max_chunk=max_chunk, overlap=2.0)

    if len(chunks) == 1:
        return _separate_audio(model, processor, device, segment_audio, sr)

    # Process in chunks and stitch
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
        # Handle both batched (2D) and unbatched (1D) output shapes
        if target.ndim >= 2:
            target = target[0]
        return target.cpu()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

- [ ] **Step 3: Add process_all_segments function**

```python
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
```

- [ ] **Step 4: Wire into main()**

Complete the `main()` function:

```python
    # Stage 3: Separate with SAM Audio
    model, processor = load_sam_audio(args.model_size, device)
    process_all_segments(
        model, processor, device, wav_path, segments,
        output_dir, args.max_chunk, args.force
    )
```

- [ ] **Step 5: Run the full pipeline on the YouTube clip**

Run: `python split_dialogue.py "https://www.youtube.com/watch?v=II1x9ptMZag"`
Expected: full pipeline runs — download (cached), diarize (cached manifest), separate each segment, output grouped WAVs.

- [ ] **Step 6: Verify output structure**

Run: `find output/II1x9ptMZag -type f | sort`
Expected output should show:
- `output/II1x9ptMZag/diarization.json`
- `output/II1x9ptMZag/raw_audio.wav`
- `output/II1x9ptMZag/speaker_0/segment_001_*.wav` (one or more per speaker directory)
- Multiple `speaker_N/` directories matching the number of speakers in `diarization.json`

Spot-check: open one WAV in a player — should contain clean voice with no background music.

- [ ] **Step 7: Commit**

```bash
git add split_dialogue.py
git commit -m "feat: stage 3 — SAM Audio voice separation with chunking and resume"
```

---

### Task 8: Final Polish & Integration Test

**Files:**
- Modify: `split_dialogue.py`

- [ ] **Step 1: Add progress summary at end of main()**

After `process_all_segments`, print a summary:

```python
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
```

- [ ] **Step 2: Run full pipeline end to end**

Run: `python split_dialogue.py "https://www.youtube.com/watch?v=II1x9ptMZag" --model-size large`
Expected: complete run with summary output showing all speakers and segment counts.

- [ ] **Step 3: Run all unit tests**

Run: `pytest tests/test_helpers.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add split_dialogue.py
git commit -m "feat: final polish — progress summary and integration verified"
```
