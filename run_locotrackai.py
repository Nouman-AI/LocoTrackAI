import inspect
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
import boxmot

# =========================
# USER SETTINGS
# =========================
# Replace these placeholder paths with your own local file paths before running.
VIDEO_PATH = r"/path/to/input_video.mp4"
YOLO_WEIGHTS_PATH = r"/path/to/LocoTrackAI-PreTrained-Model.pt"
REID_WEIGHTS_PATH = r"/path/to/osnet_x0_25_msmt17.pt"
DEEPOCSORT_CONFIG_PATH = r"configs/deepocsort.yaml"

OUTPUT_DIR = "results"
DEVICE = "cuda:0"   # Change to "cpu" if needed

IMG_SIZE = 1056
YOLO_CONF = 0.05
YOLO_IOU = 0.05
SAVE_XLSX = True
SAVE_CSV = True
# =========================


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def filter_kwargs_for_ctor(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def create_deepocsort_tracker(config_path, reid_weights_path, device):
    cfg = load_yaml_config(config_path)

    # Normalize YAML key names if needed for the BoxMOT constructor.
    if "iou_thresh" in cfg and "iou_threshold" not in cfg:
        cfg["iou_threshold"] = cfg.pop("iou_thresh")
    if "conf" in cfg and "det_thresh" not in cfg:
        cfg["det_thresh"] = cfg.pop("conf")

    runtime_args = {
        "model_weights": Path(reid_weights_path),
        "device": device,
        "fp16": True,
    }

    cfg.update(runtime_args)

    TrackerCls = getattr(boxmot, "DeepOCSORT")
    cfg = filter_kwargs_for_ctor(TrackerCls, cfg)
    return TrackerCls(**cfg)


def main():
    video_path = Path(VIDEO_PATH)
    yolo_weights_path = Path(YOLO_WEIGHTS_PATH)
    reid_weights_path = Path(REID_WEIGHTS_PATH)
    deepocsort_config_path = Path(DEEPOCSORT_CONFIG_PATH)

    if str(video_path).startswith("/path/to"):
        raise ValueError("Please edit VIDEO_PATH in run_locotrackai.py before running.")
    if str(yolo_weights_path).startswith("/path/to"):
        raise ValueError("Please edit YOLO_WEIGHTS_PATH in run_locotrackai.py before running.")
    if str(reid_weights_path).startswith("/path/to"):
        raise ValueError("Please edit REID_WEIGHTS_PATH in run_locotrackai.py before running.")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(yolo_weights_path))
    tracker = create_deepocsort_tracker(str(deepocsort_config_path), str(reid_weights_path), DEVICE)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = output_dir / "tracked_video.mp4"
    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            device=DEVICE,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for (x1, y1, x2, y2, score, cls_id) in r.boxes.data.tolist():
                detections.append([x1, y1, x2, y2, score, cls_id])

        tracks = None
        if detections:
            det_array = np.array(detections, dtype=np.float32)
            tracks = tracker.update(det_array, frame)

        if tracks is not None and len(tracks) > 0:
            tracks = np.asarray(tracks)

            for row in tracks:
                x1, y1, x2, y2, track_id = row[:5]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                rows.append({
                    "Frame": frame_idx,
                    "TrackID": int(track_id),
                    "X": float(cx),
                    "Y": float(cy),
                    "X1": float(x1),
                    "Y1": float(y1),
                    "X2": float(x2),
                    "Y2": float(y2),
                })

                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {int(track_id)}",
                    (x1i, max(20, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    if SAVE_CSV:
        df.to_csv(output_dir / "tracked_positions.csv", index=False)
    if SAVE_XLSX:
        df.to_excel(output_dir / "tracked_positions.xlsx", index=False)

    print(f"Done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
