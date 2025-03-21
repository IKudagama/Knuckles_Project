from ultralytics import YOLO

model = YOLO('runs/detect/yolov8l_100epochs/weights/best.pt')

results = model.predict('testings/test27.jpg', save=True)
print(results[0])


for box in results[0].boxes:
    print(box)