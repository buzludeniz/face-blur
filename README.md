# Face Blur

Real-time face detection and blurring from a webcam feed, in Python with
OpenCV. It finds frontal and profile faces with Haar cascades and blurs each
region live, with keyboard controls to switch the blur style and strength.

## How it works

1. **Capture.** `modules/camera.py` opens the default webcam at 640×480 / 30 fps.
2. **Detect.** `modules/detector.py` runs OpenCV's frontal-face and
   profile-face Haar cascades on a downscaled greyscale frame. The profile
   cascade also runs on a mirrored frame, so faces turned either way are found.
3. **Merge.** `modules/processor.py` merges overlapping detections and pads each
   box so the blur covers the whole face.
4. **Blur.** `modules/blur.py` applies a box blur, Gaussian blur or
   pixelation to each region.
5. **Frame skipping.** Detection runs every few frames (`FRAME_SKIP`) and the
   last known boxes are reused in between, which keeps the preview smooth.

All tunables (camera size, cascade parameters, blur strength and type,
padding, overlap threshold, detection scale) live in `config.py`.

## Controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `b` | Cycle blur type: box → gaussian → pixelate |
| `p` | Toggle profile-face detection |
| `+` / `-` | Increase / decrease blur strength |

The preview window shows the current blur type, strength, face count and
whether profile detection is on.

## Project layout

```
main.py        entry point
app.py         FaceBlurApp: main loop, key handling, on-screen info
config.py      settings
modules/
  camera.py    webcam capture
  detector.py  Haar cascade detection (frontal + profile)
  processor.py overlap merging and padding
  blur.py      box / gaussian / pixelate effects
```

## Running locally

Requires Python 3 and a webcam.

```bash
pip install -r requirements.txt
python main.py
```
