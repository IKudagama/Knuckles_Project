import os
import sys
import json
import numpy as np
import skimage.io
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

# Root directory of the project
ROOT_DIR = "D:/Demo Project/Tree Detection"
sys.path.append(ROOT_DIR)

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    """Configuration for training on the custom dataset."""
    NAME = "plants"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + plant1 + plant2
    STEPS_PER_EPOCH = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Reduce training ROIs (helps with memory)
    TRAIN_ROIS_PER_IMAGE = 32  # Default: 200 (too high for 2GB GPU)
    POST_NMS_ROIS_TRAINING = 200  # Default: 1000
    POST_NMS_ROIS_INFERENCE = 100  # Default: 2000

    # Reduce mask dimensions
    MASK_SHAPE = [28, 28]  # Default: [56, 56]

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom dataset."""
        self.add_class("plants", 1, "plant1")
        self.add_class("plants", 2, "plant2")
        assert subset in ["train", "val"]
        
        # Path to annotations JSON file
        annotations_path = os.path.join(dataset_dir, subset, f"{subset}.json")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotation file not found: {annotations_path}")
        
        # Load annotations
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # Get polygons and class names
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'].get('names') for s in a['regions'] if s['region_attributes'].get('names')]
            
            print("objects:", objects)
            num_ids = []
            for obj in objects:
                if obj == "plant1":
                    num_ids.append(1)
                elif obj == "plant2":
                    num_ids.append(2)
                else:
                    print(f"Warning: Unknown object class '{obj}', skipping.")
            
            # Load image
            image_path = os.path.join(dataset_dir, subset, a['filename'])
            if not os.path.exists(image_path):
                print(f"[WARNING] Image not found: {image_path}, skipping.")
                continue
            
            try:
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                
                self.add_image(
                    "plants",
                    image_id=a['filename'],
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons,
                    num_ids=num_ids
                )
            except Exception as e:
                print(f"[ERROR] Failed to load image {image_path}: {str(e)}")
                continue

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance_count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        
        # Handle non-plants sources
        if info["source"] != "plants":
            return super(self.__class__, self).load_mask(image_id)
        
        # Verify we have polygons and num_ids
        if not info.get("polygons") or not info.get("num_ids"):
            print(f"Warning: No polygons or num_ids for image {image_id}")
            return np.zeros([info["height"], info["width"], 0], dtype=np.uint8), np.array([], dtype=np.int32)
        
        # Ensure num_ids is a list/array and matches polygon count
        num_ids = np.array(info['num_ids']).flatten()
        if len(num_ids) != len(info["polygons"]):
            print(f"Warning: Mismatch between num_ids ({len(num_ids)}) and polygons ({len(info['polygons'])}) for image {image_id}")
            # Use minimum of the two to avoid dimension mismatch
            valid_count = min(len(num_ids), len(info["polygons"]))
            num_ids = num_ids[:valid_count]
        
        # Create mask array
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            try:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                # Clip coordinates to image dimensions
                rr = np.clip(rr, 0, info["height"]-1)
                cc = np.clip(cc, 0, info["width"]-1)
                mask[rr, cc, i] = 1
            except Exception as e:
                print(f"Warning: Failed to create mask for polygon {i} in image {image_id}: {str(e)}")
                continue
        
        return mask, num_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "plants":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset
    dataset_train = CustomDataset()
    dataset_train.load_custom("D:/Demo Project/Tree Detection/dataset", "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom("D:/Demo Project/Tree Detection/dataset", "val")
    dataset_val.prepare()

    # Training schedule
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads')

if __name__ == '__main__':
    # Configurations
    config = CustomConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    if not os.path.exists(COCO_WEIGHTS_PATH):
        utils.download_trained_weights(COCO_WEIGHTS_PATH)

    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Train the model
    train(model)