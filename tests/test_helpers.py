# tests/test_helpers.py
"""Tests for pure helper functions in split_dialogue.py."""
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
    assert len(chunks) >= 2
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 20.0
    assert chunks[-1][1] == 35.0
    for i in range(1, len(chunks)):
        overlap = chunks[i-1][1] - chunks[i][0]
        assert overlap >= 1.9


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
    result = crossfade_stitch([chunk1, chunk2], overlap_samples=overlap_samples)
    expected_len = len(chunk1) + len(chunk2) - overlap_samples
    assert len(result) == expected_len
