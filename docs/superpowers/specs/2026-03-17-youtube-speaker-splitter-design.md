# YouTube Speaker Splitter — Design Spec

## Problem

Given a YouTube URL (e.g., a movie clip with multiple characters speaking over background music), produce clean WAV files of each character's dialogue — no background score, split by speaker.

## Pipeline

Three stages: **Download → Diarize → Separate**

```
YouTube URL
  → yt-dlp (extract audio as WAV at native sample rate)
  → pyannote speaker-diarization-3.1 (who speaks when)
  → SAM Audio large/base (isolate voice from background score per segment)
  → grouped WAV files per speaker cluster
```

## Stage 1: Download

- `yt-dlp` extracts audio from the given YouTube URL
- Output: highest available sample rate, mono WAV (pyannote auto-resamples to 16kHz internally; SAM Audio's processor resamples to 24kHz internally)
- Saved to `output/<video_id>/raw_audio.wav`
- If yt-dlp fails (age-gate, geo-block, copyright takedown), print yt-dlp's stderr, clean up any partial download, and exit with a clear error distinct from HF auth failures

## Stage 2: Diarize

- pyannote `speaker-diarization-3.1` identifies speaker turns
- Produces a list of `(start_sec, end_sec, speaker_label)` segments
- Merge adjacent segments from same speaker if gap < 0.5s to avoid micro-splits
- Minimum segment duration: 0.3s (discard shorter fragments)
- Manifest written **after filtering** — only segments that will be processed appear in the manifest (no orphan IDs)
- Export `output/<video_id>/diarization.json` manifest

### Manifest format

```json
{
  "video_id": "II1x9ptMZag",
  "title": "K2SO Best scenes: Star Wars Rogue One",
  "segments": [
    {"id": 1, "start": 5.2, "end": 12.1, "speaker": "speaker_0"},
    {"id": 2, "start": 12.5, "end": 28.3, "speaker": "speaker_1"}
  ]
}
```

## Stage 3: Separate with SAM Audio

- For each diarized segment:
  1. Extract the time window from the full audio using torchaudio
  2. **Chunking**: if a segment exceeds 20 seconds, split it into overlapping chunks (20s with 2s overlap), process each chunk through SAM Audio, then crossfade-stitch the results back together. This prevents OOM on long monologues (known issue on M1 — 38s clips crash even on 64GB with MPS).
  3. Use SAM Audio (large by default, base via `--model-size base`) with text prompt `"a person speaking"` to isolate voice from background music/noise
  4. Save `result.target` as the cleaned WAV file
- **Prompt limitation**: the same generic prompt is used for all speakers. SAM Audio separates voice from non-voice — it does not distinguish between different speakers. Speaker identity comes from the diarization stage, not from SAM Audio.
- Group output files by speaker cluster
- Process segments sequentially to manage memory on M1 (14.9GB model + audio buffers)
- **Resumability**: before processing a segment, check if its output WAV already exists and skip it. This allows resuming after a crash without re-processing completed segments.

## Output Structure

```
output/<video_id>/
  raw_audio.wav
  diarization.json
  speaker_0/
    segment_001_0m05s-0m12s.wav
    segment_003_0m30s-0m38s.wav
  speaker_1/
    segment_002_0m12s-0m28s.wav
  ...
```

## CLI Interface

```
python split_dialogue.py <youtube_url> [options]

Options:
  --model-size      large|base (default: large)
  --output-dir      output directory (default: ./output)
  --min-segment     minimum segment duration in seconds (default: 0.3)
  --merge-gap       max gap to merge same-speaker segments (default: 0.5)
  --max-chunk       max segment length before chunking, in seconds (default: 20)
  --device          cpu|cuda (default: auto-detect)
  --force           re-process all segments even if output exists
```

## Dependencies

- `yt-dlp` — YouTube audio download
- `torch`, `torchaudio` — tensor ops and audio I/O
- `sam_audio` — Meta's SAM Audio model (source separation)
- `pyannote.audio` — speaker diarization
- Standard lib: `argparse`, `json`, `pathlib`, `subprocess`

## Authentication

Both SAM Audio and pyannote require:
1. A HuggingFace account
2. Accepting model terms on each model's HF page
3. `HF_TOKEN` environment variable or `huggingface-cli login`

The script checks for auth upfront and exits with a clear message if missing.

## Hardware Notes

- SAM Audio large: ~14.9GB checkpoint → fits in 32GB M1 MacBook Pro
- **CPU only** — MPS support for SAM Audio is broken upstream (facebookresearch/sam-audio#29) and requires PyTorch nightly + manual patches. Not worth the fragility. The `--device` flag supports `cpu` and `cuda` only.
- Processing is sequential per-segment to keep memory stable
- Segments longer than 20s are chunked to prevent OOM
- Expect 1-2+ hours for a multi-minute clip with large model
- Base model will be significantly faster

## Single File

Everything lives in one script: `split_dialogue.py`. No package structure needed.
