# tt_object_yolo

YOLOv8-based room occupancy analyzer. Detects people, chairs, and benches in a room image and estimates seating capacity, available space, and overall occupancy status.

## Features

- Uses YOLOv8 for object detection (`person`, `chair`, `bench`)
- Estimates seating capacity and available space
- Annotates image with bounding boxes and labels
- Outputs human-readable occupancy summary

## Requirements

- Python 3.8+
- `ultralytics`
- `opencv-python`
- `supervision`
- `torch`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py <path_to_image> [--model yolov8x.pt] [--no-show] [--no-save]
```

### Arguments

- `<path_to_image>`: Image file to analyze (`.jpg`, `.jpeg`, `.png`)
- `--model`: Optional. YOLOv8 model file. Defaults to `yolov8x.pt`
- `--no-show`: Don’t display annotated image after saving
- `--no-save`: Don’t save or display annotated image at all

## Output

- Console: Counts people, chairs, benches, seating capacity, available space, and interpretation
- Image: Saved as `<original_name>_annotated.<ext>` unless `--no-save` is set

## Example

```bash
python main.py room.jpg
```
