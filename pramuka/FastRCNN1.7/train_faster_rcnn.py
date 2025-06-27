
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils import CocoDetectionTransform
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 17  # 13 trees + background
num_epochs = 100

def get_transform(train=True):
    transforms = [T.ToTensor()]
    if train:
        transforms += [
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomRotation(10),
        ]
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# Load datasets
train_dataset = CocoDetectionTransform(
    img_folder='Trees/train/',
    ann_file='Trees/train/train.json',
    transforms=get_transform(train=True)
)
val_dataset = CocoDetectionTransform(
    img_folder='Trees/valid/',
    ann_file='Trees/valid/valid.json',
    transforms=get_transform(train=False)
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Load model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Tracking
train_losses = []
val_losses = []
val_maps = []
val_precisions = []
val_recalls = []

# Evaluate function
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)

            outputs = model(images)

            for output, target in zip(outputs, targets):
                all_preds.append(output)
                all_targets.append(target)

    # You can implement your own mAP/precision/recall calculation or return dummy metrics:
    return 0.0, 0.0, 0.0, 0.0  # val_loss, mAP, precision, recall

# Training loop
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
    val_loss, mAP, precision, recall = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_maps.append(mAP)
    val_precisions.append(precision)
    val_recalls.append(recall)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - mAP@0.5: {mAP:.3f} - Prec: {precision:.3f} - Rec: {recall:.3f}")

torch.save(model.state_dict(), 'faster_rcnn_resnet50_tree_detector.pth')

# Plotting
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Box Loss')
plt.plot(val_losses, label='Val Box Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_maps, label='mAP@0.5')
plt.plot(val_precisions, label='Precision')
plt.plot(val_recalls, label='Recall')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
