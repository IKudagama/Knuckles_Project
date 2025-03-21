from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(
    data = "dataset_garden/data.yaml",
    epochs = 100,
    batch = 8,
    imgsz = 640,
    device = "cpu"
)