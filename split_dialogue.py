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


def main():
    args = parse_args()
    check_auth()
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Model: facebook/sam-audio-{args.model_size}")
    print(f"URL: {args.url}")


if __name__ == "__main__":
    main()
