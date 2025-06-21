# import torch
# import torchvision
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from PIL import Image, ImageDraw, ImageFont
# import torchvision.transforms as T
# import os

# # ----- Load the model -----
# num_classes = 2  # 1 class (tree) + background
# model = fasterrcnn_resnet50_fpn(weights=None)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# model.load_state_dict(torch.load("Tree_best.pth", map_location="cpu"))
# model.eval()

# # ----- Load and transform image -----
# transform = T.Compose([
#     T.ToTensor()
# ])

# image_path = "test.jpg"
# output_path = "result_1.jpg"

# image = Image.open(image_path).convert("RGB")
# image_tensor = transform(image).unsqueeze(0)  # add batch dimension

# # ----- Run inference -----
# with torch.no_grad():
#     predictions = model(image_tensor)[0]

# # ----- Draw boxes -----
# draw = ImageDraw.Draw(image)
# font = ImageFont.load_default()  # for drawing text

# for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
#     if score > 0.5:  # confidence threshold
#         x1, y1, x2, y2 = box.tolist()
#         draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
#         draw.text((x1, y1 - 10), f"Tree: {score:.2f}", fill="green", font=font)

# # ----- Save result -----
# image.save(output_path)
# print(f"Detection complete. Saved output to {output_path}")


import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import os

# ----- Class labels -----
category_names = {
    0: "background",
    1: "Tree1",
    2: "Tree2",
    3: "Tree3",
    4: "Tree4",
    5: "Tree5",
}


# ----- Load the model -----
num_classes = 6  # 1 class (Tree) + background
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("faster_rcnn_tree_detector.pth", map_location="cpu"))
model.eval()

# ----- Load and transform image -----
transform = T.Compose([
    T.ToTensor()
])

image_path = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\FastRCNN1.5\test\test12.jpg"
output_path = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\FastRCNN1.5\result\result12.jpg"

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# ----- Run inference -----
with torch.no_grad():
    predictions = model(image_tensor)[0]

# ----- Draw boxes -----
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # For drawing text

for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
    if score > 0.5:  # Confidence threshold
        x1, y1, x2, y2 = box.tolist()
        label_id = label.item()
        label_name = category_names.get(label_id, "Unknown")
        draw.rectangle([x1, y1, x2, y2], outline="green", width=5)
        draw.text((x1, y1 - 10), f"{label_name}: {score:.2f}", fill="black", font=font)

# ----- Save result -----
image.save(output_path)
print(f"Detection complete. Saved output to {output_path}")
