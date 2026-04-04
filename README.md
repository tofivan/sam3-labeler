# SAM3 Labeler

Interactive annotation tool powered by Meta SAM 3 — click, box-select, or text-prompt to segment objects, then export in OBB / YOLO-Seg / PNG Mask / COCO JSON formats.

> **v1 (Gradio)** is available at tag [`v1.0`](../../tree/v1.0).
> **v2 (PyQt6)** is the current version — a full desktop rewrite with real-time rendering.

## Installation

```bash
# 1. Install PyTorch (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install dependencies
pip install PyQt6 opencv-python numpy ultralytics
```

### SAM 3 Model

Download `sam3.pt` (~3.4 GB) from [Hugging Face](https://huggingface.co/facebook/sam3) (requires Meta approval).

## Quick Start

```bash
python main.py --model /path/to/sam3.pt
python main.py --model sam3.pt --images ./images --output ./output
```

## Three Annotation Modes

| Mode | Shortcut | Description |
|------|----------|-------------|
| Click | `1` | Click object center, SAM auto-segments |
| Box Select | `2` | Drag a rectangle, SAM segments within |
| Text Prompt | — | Type object name, SAM finds all matches |

## Four Output Formats

| Format | Path | Usage |
|--------|------|-------|
| OBB | `labels/*.txt` | YOLO OBB training |
| YOLO-Seg | `labels_seg/*.txt` | Instance segmentation |
| PNG Mask | `masks/*.png` | Semantic segmentation |
| COCO JSON | `coco_annotations.json` | COCO-format training |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Previous / Next image (auto-save) |
| `1` `2` `3` | Switch mode: Click / Box / Select |
| `Delete` | Delete selected annotations |
| `Ctrl+S` | Save |
| `Ctrl+A` | Select all |
| `Esc` | Deselect |
| `F` | Fit to window |
| Scroll wheel | Zoom (centered on cursor) |
| Right-click drag | Pan |
| Space + left-click drag | Pan |
| Middle-click drag | Pan |
| Double-click | Fit to window |

## v2 Highlights (vs v1 Gradio)

### Performance

| Action | Gradio (v1) | PyQt6 (v2) |
|--------|-------------|------------|
| Click → redraw | 300-500 ms | < 5 ms |
| Hover highlight | Not supported | < 3 ms |
| Box drag | No feedback | Real-time |
| SAM inference | UI freezes | Background thread |
| Zoom / Pan | Not supported | Scroll + drag |

### New Features
- QPainter vector rendering (replaces server-side image encoding)
- Right-click / Space+click / Middle-click pan for laptops
- Fit-to-window button + `F` key
- Output format checkboxes (OBB / Seg / Mask / COCO)
- Polygon simplification slider (0.001~0.020)
- Overlap threshold slider (0%~50%)
- Class delete button, color indicators in annotation list
- Dark sci-fi theme, scrollable sidebar

## Project Structure

```
sam3-labeler/
├── main.py              # Entry point
├── requirements.txt
├── docs/
│   └── USER_MANUAL_zh-TW.md
├── core/
│   ├── state.py         # LabelingState
│   ├── utils.py         # Coordinate transforms, overlap detection
│   ├── io_manager.py    # Config, progress, label I/O
│   └── sam_engine.py    # SAM 3 model wrapper
└── ui/
    ├── canvas.py        # QPainter vector canvas
    └── main_window.py   # Main window + control panels
```

## Documentation

[中文使用手冊](docs/USER_MANUAL_zh-TW.md) | [English User Manual](docs/USER_MANUAL_en.md)

## Acknowledgments

This project is supported by the **Ocean Conservation Administration, Ocean Affairs Council** (海洋委員會海洋保育署).

## License

MIT
