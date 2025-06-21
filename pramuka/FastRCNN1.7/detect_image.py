import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet101_fpn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

# ----- Class labels -----
category_names = {
    0: "background",
    1: "Tree1", 2: "Tree2", 3: "Tree3", 4: "Tree4", 5: "Tree5",
    6: "Tree6", 7: "Tree7", 8: "Tree8", 9: "Tree9", 10: "Tree10",
    11: "Tree11", 12: "Tree12", 13: "Tree13"
}

# Load model
num_classes = 14
model = fasterrcnn_resnet101_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("faster_rcnn_resnet101_tree_detector.pth", map_location="cpu"))
model.eval()

# Load and transform image
transform = T.Compose([T.ToTensor()])
image_path = "test.jpg"
output_path = "result_resnet101.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)[0]

# Draw results
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
    if score > 0.5:
        x1, y1, x2, y2 = box.tolist()
        label_name = category_names.get(label.item(), "Unknown")
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 10), f"{label_name}: {score:.2f}", fill="black", font=font)

image.save(output_path)
print(f"Detection complete. Saved output to {output_path}")
