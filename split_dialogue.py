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
