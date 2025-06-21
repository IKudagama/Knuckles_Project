import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils import CocoDetectionTransform
import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.ToTensor()
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

# Load datasets
train_dataset = CocoDetectionTransform(
    img_folder='dress_detect/train/images',
    ann_file='dress_detect/train/train.json',
    transforms=get_transform()
)

val_dataset = CocoDetectionTransform(
    img_folder='dress_detect/valid/images',
    ann_file='dress_detect/valid/valid.json',
    transforms=get_transform()
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Load model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 3  # 1 class (tree) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Train setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}")

# Save model
torch.save(model.state_dict(), 'faster_rcnn_tree_detector.pth')
