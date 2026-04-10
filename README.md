# LocoTrackAI

This repository provides a simple example script for running LocoTrackAI-based mosquito tracking using YOLO and DeepOCSORT.

## Files
- `run_locotrackai.py` — example tracking script using YOLO and DeepOCSORT
- `configs/deepocsort.yaml` — DeepOCSORT configuration used by the example script
- `configs/*.yaml` — configuration files for all evaluated trackers, provided for hyperparameter transparency
- `requirements.txt` — Python package requirements

## Before running
Open `run_locotrackai.py` and replace these placeholder paths with your own local file paths:
- `VIDEO_PATH`
- `YOLO_WEIGHTS_PATH`
- `REID_WEIGHTS_PATH`

The DeepOCSORT config path is already set to:
- `configs/deepocsort.yaml`

## Main settings used
- YOLO image size: `1056`
- YOLO confidence threshold: `0.05`
- YOLO IoU threshold: `0.05`
- Tracker: `DeepOCSORT`

## Run
```bash
python run_locotrackai.py
```

## Output
The script saves:
- `results/tracked_video.mp4`
- `results/tracked_positions.csv`
- `results/tracked_positions.xlsx`

## Notes
- The example script is intentionally simple and focuses on the standard LocoTrackAI run path using YOLO and DeepOCSORT.
- The `configs` folder includes configuration files for all evaluated trackers so hyperparameter settings can be directly inspected.
- Do not upload large result videos or generated result folders back into the repository.
