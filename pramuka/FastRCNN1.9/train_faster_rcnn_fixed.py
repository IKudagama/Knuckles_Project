import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import CocoDetectionTransform, get_transform
from pycocotools.cocoeval import COCOeval
import json
import tempfile

# ---- Config ----
data_dir = "Trees"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")
train_ann = os.path.join(train_dir, "train.json")
val_ann = os.path.join(val_dir, "valid.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 17
num_epochs = 100
batch_size = 8
learning_rate = 0.005

# ---- Datasets & DataLoaders ----
train_dataset = CocoDetectionTransform(img_folder=train_dir, ann_file=train_ann, transforms=get_transform(train=True))
val_dataset = CocoDetectionTransform(img_folder=val_dir, ann_file=val_ann, transforms=get_transform(train=False))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ---- Model ----
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# ---- Optimizer ----
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)

# ---- Evaluation ----
def evaluate(model, data_loader, device):
    model.eval()
    coco_gt = data_loader.dataset.coco
    coco_results = []
    image_ids = []
    val_loss_total = 0.0
    count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Validation loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss_total += losses.item()
            count += 1

            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                image_ids.append(image_id)
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score)
                    })

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
        json.dump(coco_results, tmp_file)
        tmp_file.flush()
        coco_dt = coco_gt.loadRes(tmp_file.name)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = list(set(image_ids))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    val_loss = val_loss_total / count
    map50 = coco_eval.stats[1]
    precision = coco_eval.stats[0]
    recall = coco_eval.stats[8]

    return val_loss, map50, precision, recall

# ---- Training ----
train_losses = []
val_losses = []
mAPs = []
precisions = []
recalls = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    train_loss = running_loss / len(train_loader)
    val_loss, mAP, precision, recall = evaluate(model, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    mAPs.append(mAP)
    precisions.append(precision)
    recalls.append(recall)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - mAP@0.5: {mAP:.3f} - Prec: {precision:.3f} - Rec: {recall:.3f}")

# ---- Save Model ----
torch.save(model.state_dict(), "faster_rcnn_tree_detector.pth")

# ---- Plotting ----
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(epochs, mAPs, label="mAP@0.5")
plt.plot(epochs, precisions, label="Precision")
plt.plot(epochs, recalls, label="Recall")
plt.legend()
plt.title("Evaluation Metrics")
plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
