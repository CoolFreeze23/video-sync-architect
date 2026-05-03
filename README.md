# Video Sync Architect

Desktop app (PyQt6) for aligning a **target** video to a **primary** reference using perceptual hashing (pHash), with optional **audio VAD** and **scene-cut** verification, **multi-reference** matching, and **segmented** export when offsets vary over time (e.g. different bumper lengths).

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` on your `PATH`
- Dependencies: see `video_sync_architect/requirements.txt`

## Install

```powershell
cd path\to\this\repo
pip install -r video_sync_architect/requirements.txt
```

## Run

From the **repository root** (the folder that contains `video_sync_architect/`):

```powershell
python -m video_sync_architect.main
```

On Windows you can also use `pythonw -m video_sync_architect.main` for a console-free window.

## Layout

This repository contains the `video_sync_architect` Python package only. If you cloned from a workspace that also had other projects in the same folder, those extra files are not tracked by Git.
