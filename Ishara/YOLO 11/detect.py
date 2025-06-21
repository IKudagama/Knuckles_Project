from ultralytics import YOLO

model = YOLO('runs/detect/train14/weights/best.pt')

results = model.predict('test_images/test12.jpeg', save=True)
print(results[0])


for box in results[0].boxes:
    print(box)