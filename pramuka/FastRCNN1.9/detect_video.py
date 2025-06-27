import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet101_fpn
import torchvision.transforms as T
import cv2
from PIL import Image

# Load model
num_classes = 14
model = fasterrcnn_resnet101_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("faster_rcnn_resnet101_tree_detector.pth", map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class_names = [
    "background", "Tree1", "Tree2", "Tree3", "Tree4", "Tree5", "Tree6",
    "Tree7", "Tree8", "Tree9", "Tree10", "Tree11", "Tree12", "Tree13"
]

input_path = "input_video.mp4"
output_path = "output_video_resnet101.mp4"
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
transform = T.ToTensor()
font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)[0]

    for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            label_name = class_names[label] if label < len(class_names) else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10), font, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Detection complete. Saved video to {output_path}")
