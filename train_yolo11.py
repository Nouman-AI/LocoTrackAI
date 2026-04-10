from ultralytics import YOLO


def train_model():
    data_yaml_path = r"/path/to/mosquitoconfig.yml"

    # Initialize model from official pretrained YOLO11 nano weights
    model = YOLO("yolo11n.pt")

    # Train model
    model.train(
        data=data_yaml_path,
        imgsz=1056,
        epochs=1000,
        workers=4,
        patience=50,
        batch=16,
        device=0,
    )


if __name__ == "__main__":
    train_model()
