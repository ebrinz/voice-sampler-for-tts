<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ebrinz/voice-sampler-for-tts/master/banner.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/ebrinz/voice-sampler-for-tts/master/banner.svg">
    <img alt="Voice Sampler for TTS" src="https://raw.githubusercontent.com/ebrinz/voice-sampler-for-tts/master/banner.svg" width="100%">
  </picture>
</p>

Extract clean character dialogue from YouTube clips — no background music, split by speaker. Outputs WAV files ready for voice cloning (e.g., Chatterbox TTS).

## What It Does

```
YouTube URL
  → Downloads audio
  → SAM Audio removes background music/score
  → Pyannote detects who speaks when
  → Clean WAV files per speaker + Audacity labels for fine-tuning
```

## Prerequisites

### 1. Python Environment

Requires Python 3.11+.

```bash
# Clone the repo
git clone <this-repo>
cd voice-sampler-for-tts

# Install core dependencies
pip install torch torchaudio pyannote.audio yt-dlp pytest

# Install SAM Audio and its dependencies (not on PyPI)
pip install --no-deps git+https://github.com/facebookresearch/sam-audio.git
pip install --no-deps "perception-models @ git+https://github.com/facebookresearch/perception_models@unpin-deps"
pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
pip install --no-deps git+https://github.com/lematt1991/CLAP.git
pip install dacvae torchdiffeq torchvision torchlibrosa pytorchvideo
pip install braceexpand wget webdataset "protobuf>=5.0,<7.0"
```

### 2. macOS Patches (Apple Silicon)

SAM Audio has dependencies that don't work out of the box on macOS. You need to patch three files to make `xformers` and `torchcodec` optional:

**Find your site-packages:**
```bash
python -c "import site; print(site.getsitepackages()[0])"
```

**Patch `core/transformer.py`** — replace line 13:
```python
# Before:
from xformers.ops import AttentionBias, fmha

# After:
try:
    from xformers.ops import AttentionBias, fmha
except ImportError:
    fmha = None
    AttentionBias = None
```

**Patch `core/probe.py`** — replace the xformers import:
```python
# Before:
from xformers.ops import fmha

# After:
try:
    from xformers.ops import fmha
except ImportError:
    fmha = None
```

**Patch `core/profiling.py`** — replace the xformers imports:
```python
# Before:
import xformers.profiler
from torch.profiler.profiler import profile
from xformers.profiler import MemSnapshotsProfiler, PyTorchProfiler

# After:
from torch.profiler.profiler import profile
try:
    import xformers.profiler
    from xformers.profiler import MemSnapshotsProfiler, PyTorchProfiler
except ImportError:
    xformers = None
    MemSnapshotsProfiler = None
    PyTorchProfiler = None
```

**Patch `core/audio_visual_encoder/transforms.py`** — replace the torchcodec import:
```python
# Before:
from torchcodec.decoders import AudioDecoder, VideoDecoder

# After:
try:
    from torchcodec.decoders import AudioDecoder, VideoDecoder
except (ImportError, RuntimeError):
    AudioDecoder = None
    VideoDecoder = None
```

**Patch `sam_audio/model/model.py`** — replace line 94:
```python
# Before:
self.visual_ranker = create_ranker(cfg.visual_ranker)

# After:
try:
    self.visual_ranker = create_ranker(cfg.visual_ranker)
except (AssertionError, ImportError):
    self.visual_ranker = None
```

**Patch `sam_audio/model/base.py`** — make `_from_pretrained` compatible with newer `huggingface_hub`. Change the method signature so all parameters after `model_id` have defaults, and remove `proxies` and `resume_download` from the `snapshot_download()` call:
```python
@classmethod
def _from_pretrained(
    cls,
    *,
    model_id: str,
    cache_dir: str = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    resume_download: bool = False,
    local_files_only: bool = False,
    token: Union[str, bool, None] = None,
    map_location: str = "cpu",
    strict: bool = True,
    revision: Optional[str] = None,
    **model_kwargs,
):
    if os.path.isdir(model_id):
        cached_model_dir = model_id
    else:
        download_kwargs = dict(
            repo_id=model_id,
            revision=cls.revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
        if proxies is not None:
            download_kwargs["proxies"] = proxies
        cached_model_dir = snapshot_download(**download_kwargs)
    # ... rest of method unchanged
```

### 3. HuggingFace Authentication

Both SAM Audio and pyannote are gated models. You need to:

1. Create a [HuggingFace account](https://huggingface.co/join)
2. Accept the model terms:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [facebook/sam-audio-base](https://huggingface.co/facebook/sam-audio-base) (and/or `sam-audio-large`)
3. Log in:
   ```bash
   huggingface-cli login
   ```

### 4. Install Audacity

Download from [audacityteam.org](https://www.audacityteam.org/download/) or:
```bash
brew install --cask audacity
```

---

## Step 1: Run the Script

```bash
python split_dialogue.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-size` | `large` | `large` (best quality, 14.9GB, slow) or `base` (faster, smaller) |
| `--output-dir` | `./output` | Where to save output |
| `--min-segment` | `0.3` | Minimum segment duration in seconds |
| `--merge-gap` | `0.5` | Merge same-speaker segments closer than this (seconds) |
| `--max-chunk` | `20` | Max chunk length for SAM Audio processing (prevents OOM) |
| `--device` | auto | `cpu` or `cuda` |
| `--force` | off | Re-process everything, ignoring cached results |

### Example

```bash
# Fast run with base model
python split_dialogue.py "https://www.youtube.com/watch?v=II1x9ptMZag" --model-size base

# Best quality with large model (slow on CPU)
python split_dialogue.py "https://www.youtube.com/watch?v=II1x9ptMZag" --model-size large
```

### What It Produces

```
output/<video_id>/
  raw_audio.wav        # Original audio from YouTube
  clean_audio.wav      # Voice-only audio (background music removed)
  diarization.json     # Segment manifest with timestamps + speaker labels
  labels.txt           # Audacity labels file
  SPEAKER_00/          # Segments grouped by speaker
    segment_001_*.wav
    segment_002_*.wav
  SPEAKER_01/
    ...
```

### Hardware Notes

- **SAM Audio large**: ~14.9GB — fits on 32GB M1 MacBook Pro, CPU only
- **SAM Audio base**: ~3GB — much faster, still good quality
- Processing time: base model ~30 min for a 3 min clip on M1, large model ~2+ hours
- Segments >20s are automatically chunked to prevent out-of-memory crashes

---

## Step 2: Open in Audacity

1. Open Audacity
2. **File → Open** → select `output/<video_id>/clean_audio.wav`
3. **File → Import → Labels** → select `output/<video_id>/labels.txt`

You'll see the full waveform with yellow labels marking every detected speech segment:

```
|--seg001--|  |---seg002---|  |seg003|  |----seg004----|  ...
```

---

## Step 3: Review and Adjust Segments

### Navigate segments
- Click any **label text** (e.g., `seg001`) to select that region
- Press **Space** to play the selection
- Use **Ctrl+1** to zoom in, **Ctrl+3** to zoom out

### Adjust boundaries
- Hover over the **edge of a label region** — the cursor changes to a resize arrow
- **Drag** to adjust the start or end point
- Listen again to verify

### Remove unwanted segments
- Click the label text
- Press **Backspace/Delete** to remove the label
- Or right-click the label → **Delete Label**

### Add new segments you spot
- Click and drag to select a region on the waveform
- **Edit → Labels → Add Label at Selection** (or **Ctrl+B**)
- Type a name (e.g., `k2so_extra`)

### Rename segments by character
- Double-click a label text to edit it
- Rename from `seg001` to the character name (e.g., `k2so_001`, `cassian_001`)

---

## Step 4: Master the Audio

Make all segments sound consistent in volume and tone.

### Loudness Normalization
1. **Select All** — `Cmd+A` (Mac) or `Ctrl+A` (Windows)
2. **Effect → Loudness Normalization**
3. Set perceived loudness to **-16 LUFS**
4. Click **Apply**

### Compression (even out dynamics)
1. **Select All** again
2. **Effect → Compressor**
3. Settings:
   - Threshold: **-12 dB**
   - Noise Floor: **-40 dB**
   - Ratio: **4:1**
   - Attack Time: **0.2 sec**
   - Release Time: **1.0 sec**
4. Click **Apply**

### Remove remaining low rumble (optional)
1. **Select All**
2. **Effect → High-Pass Filter**
3. Frequency: **80 Hz**
4. Roll-off: **12 dB per octave**
5. Click **Apply**

### De-essing (optional, for harsh S sounds)
1. **Effect → Graphic EQ** or **Filter Curve EQ**
2. Pull down the **4kHz–8kHz** range by **2–3 dB**
3. Click **Apply**

---

## Step 5: Export Individual Clips

1. **File → Export → Export Multiple**
2. Configure:
   - Format: **WAV (Microsoft)**
   - Encoding: **Signed 16-bit PCM** (or 32-bit float for higher quality)
   - Split files based on: **Labels**
   - Name files: **Using Label/Track Name**
3. Choose your output folder
4. Click **Export**

Each labeled region becomes its own WAV file, named after the label.

---

## Step 6: Use with Chatterbox TTS

The exported WAV files are ready to use as voice references in Chatterbox:

```python
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")

# Use your best ~10s clip as the voice reference
wav = model.generate(
    "I find that answer vague and unconvincing.",
    audio_prompt_path="k2so_best_clip.wav"
)
```

**Tips for best results:**
- Pick reference clips that are **5–15 seconds** long
- Choose clips with **clear, representative speech** (no whispering, shouting, or overlapping)
- One clip per character is enough — pick the cleanest one
- If a character has a distinctive voice (like K-2SO's robotic tone), even a short 3–5s clip works well

---

## Troubleshooting

### Only 1 speaker detected
This happens with compilation clips (highlight reels, "best of" videos) where scenes jump around. For best multi-speaker detection, use clips with **continuous back-and-forth dialogue** between characters.

### Out of memory
- Use `--model-size base` instead of `large`
- Lower `--max-chunk` to `15` or `10`
- Close other applications to free RAM

### yt-dlp fails with 403
```bash
pip install --upgrade yt-dlp
```

### SAM Audio import errors on macOS
See the [macOS Patches](#2-macos-patches-apple-silicon) section above. The key issues are `xformers` (CUDA-only) and `torchcodec` (FFmpeg version mismatch) — both need try/except wrappers.

### Slow processing
- The `base` model is ~4x faster than `large` with slightly lower separation quality
- Processing is CPU-bound on Mac (no MPS support for SAM Audio)
- Each 20s chunk takes ~2–3 min on M1 with the base model
