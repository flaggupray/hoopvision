# 🏀 HoopVision

Basketball game AI analyzer — upload a game video, get a narrative timeline with jersey numbers, shot types, and player actions.

## Architecture

```
hoopvision/
├── engine/          # Python AI engine (YOLO + Pose + OCR + Classifier)
├── macos/           # macOS SwiftUI app (Liquid Glass)
├── shared/          # Jinja2 HTML templates
└── tests/           # 66 tests
```

## Quick Start

```bash
# Install
cd hoopvision
pip install -e ".[dev]"

# Download models (auto on first run, or manually)
python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt'); YOLO('yolo11n-pose.pt')"

# Analyze a video
python3 -m engine.cli run game.mp4 --home Lakers --away Warriors -o timeline.html
```

## AI Pipeline

1. **Decoder** — Adaptive frame sampling (15fps)
2. **Detector** — YOLOv11m person/ball detection + ByteTrack + optical flow
3. **Pose** — YOLOv11-pose shooting motion detection
4. **OCR** — PaddleOCR jersey number recognition
5. **Classifier** — 3-source fusion (ball / flow / pose) → 12 event types
6. **Compiler** — Dedup, sort, assemble timeline
7. **Narrator** — 80+ Chinese narrative templates
8. **Exporter** — Liquid Glass HTML + JSON

## Event Types

| Category | Events |
|----------|--------|
| Shot | 3PT made/missed, 2PT made/missed, free throw made/missed |
| Rebound | Offensive, defensive |
| Other | Assist, steal, block, timeout |

## macOS App

```bash
cd macos
swift build
bash build.sh
open HoopVision.app
```

## Tech Stack

- Python 3.11, PyTorch, Ultralytics YOLOv11
- PaddleOCR, OpenCV, Jinja2, Pydantic
- Swift 6, SwiftUI, AppKit
