import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import os

# ----- Class labels -----
category_names = {
    0: "background",
    1: "Tree1", 2: "Tree2", 3: "Tree3", 4: "Tree4", 5: "Tree5",
    6: "Tree6", 7: "Tree7", 8: "Tree8", 9: "Tree9", 10: "Tree10",
    11: "Tree11", 12: "Tree12", 13: "Tree13"
}

# ----- Paths -----
model_path = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\Official_Rep\Knuckles_Project\pramuka\FastRCNN1.9\faster_rcnn_tree_detector.pth"
input_dir = "test2"
output_dir = "results2"

# ----- Ensure output directory exists -----
os.makedirs(output_dir, exist_ok=True)

# ----- Load model -----
num_classes = 14
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ----- Transform -----
transform = T.Compose([T.ToTensor()])

# ----- Loop through 41 images -----
for i in range(201, 206):  # test1.jpg to test41.jpg
    image_path = os.path.join(input_dir, f"test{i}.jpg")
    output_path = os.path.join(output_dir, f"result{i}.jpg")

    if not os.path.exists(image_path):
        print(f"[{i}/57] Image not found: {image_path}")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[{i}/57] Failed to open {image_path}: {e}")
        continue

    image_tensor = transform(image).unsqueeze(0)

    try:
        with torch.no_grad():
            predictions = model(image_tensor)[0]
    except Exception as e:
        print(f"[{i}/57] Inference failed on {image_path}: {e}")
        continue

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        if score > 0.5:
            x1, y1, x2, y2 = box.tolist()
            label_name = category_names.get(label.item(), "Unknown")
            draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
            draw.text((x1, y1 - 10), f"{label_name}: {score:.2f}", fill="black", font=font)

    try:
        image.save(output_path)
        print(f"[{i}/57] ✅ Detection complete. Saved to {output_path}")
    except Exception as e:
        print(f"[{i}/57] ❌ Failed to save result image: {e}")

print("🎉 All done!")
