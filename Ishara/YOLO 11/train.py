from ultralytics import YOLO


def main():

    model = YOLO("yolo11n.pt")

    results = model.train(
        data = "Trees-1/data.yaml",
        epochs = 100,
        batch = 8,
        imgsz = 640,
        device = "0"

    
)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Only needed for PyInstaller apps, optional otherwise
    
    main()

