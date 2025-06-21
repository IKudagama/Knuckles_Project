import torch
import torchvision.transforms as T
from torchvision.datasets import CocoDetection

class CocoDetectionTransform(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoDetectionTransform, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetectionTransform, self).__getitem__(idx)
        image_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Convert COCO to expected format
        target = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue  # skip invalid boxes
            x1, y1 = x, y
            x2, y2 = x + w, y + h
        target.append({
            'boxes': torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
            'labels': torch.tensor([ann['category_id']], dtype=torch.int64)
        })

        
        if self._transforms:
            img = self._transforms(img)
        
        # Merge all boxes and labels
        if target:
            boxes = torch.cat([t['boxes'] for t in target], dim=0)
            labels = torch.cat([t['labels'] for t in target], dim=0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.int64)

        return img, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}
