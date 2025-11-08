import os
import json
import numpy as np
import sys
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import mrcnn.model
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_instances
from IPython.display import Image, display
from mrcnn import model as modellib

class ChickenConfig(Config):

    NAME = "Chicken"

    # GPU configuration for Colab
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  # Reduce to 1 if out of memory

    # Classes
    NUM_CLASSES = 1 + 1  # Background + chicken

    # Training
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50

    # Image settings
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Learning
    LEARNING_RATE = 0.001

    # Detection
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.3

    # ROIs
    TRAIN_ROIS_PER_IMAGE = 200
    MAX_GT_INSTANCES = 10

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

# dataset loader
class ChickenDataset(utils.Dataset):
    def load_chickens(self, dataset_dir, subset):
        # Add class
        self.add_class("chicken", 1, "chicken")

        # Path to JSON and images
        json_path = os.path.join(dataset_dir, subset, "annotations.json")
        img_dir = os.path.join(dataset_dir, subset, "images")

        # Check if files exist
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations not found: {json_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        with open(json_path) as f:
            coco = json.load(f)

        # Add images
        for img in coco["images"]:
            image_id = img["id"]
            self.add_image(
                "chicken",
                image_id=image_id,
                path=os.path.join(img_dir, img["file_name"]),
                width=img["width"], height=img["height"],
                annotations=[a for a in coco["annotations"] if a["image_id"] == image_id]
            )

    def load_mask(self, image_id):
        """Return masks and class_ids for an image."""
        info = self.image_info[image_id]
        annots = info.get("annotations", [])
        masks = []
        class_ids = []
        from pycocotools import mask as maskUtils

        for ann in annots:
            seg = ann.get("segmentation", None)
            if seg:
                # Use COCO utilities to convert
                rles = maskUtils.frPyObjects(seg, info["height"], info["width"])
                rle = maskUtils.merge(rles) if isinstance(rles, (list, tuple)) else rles
                m = maskUtils.decode(rle)
                masks.append(m)
                class_ids.append(ann["category_id"])

        if masks:
            masks = np.stack(masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        else:
            return np.empty((0, 0, 0)), np.array([], dtype=np.int32)

# Main
config = ChickenConfig()
# Root directory of the project
ROOT_DIR = os.path.abspath("/content/Mask_RCNN/mrcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
# Assuming ROOT_DIR is defined elsewhere, replace with appropriate path if not
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = os.path.join("/content/Mask_RCNN", "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# Assuming ROOT_DIR is defined elsewhere, replace with appropriate path if not
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = os.path.join("/content/Mask_RCNN", "logs")

# Training
train_dir = '/content/drive/MyDrive/ChickenDataset'
val_dir = '/content/drive/MyDrive/ChickenDataset/'

dataset_train = ChickenDataset()
dataset_train.load_chickens(train_dir, "training")
dataset_train.prepare()

dataset_val = ChickenDataset()
dataset_val.load_chickens(val_dir, "validation")
dataset_val.prepare()

# Summary LOG
print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"Training images: {len(dataset_train.image_ids)}")
print(f"Validation images: {len(dataset_val.image_ids)}")
print(f"Classes: {dataset_train.class_names}")
print("="*70)

# Load and display random samples
os.makedirs("sample", exist_ok=True)
image_ids = np.random.choice(dataset_train.image_ids, 4)
for idx, image_id in enumerate(image_ids):
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    
    # simpan ke folder sample/
    output_path = f"sample/output_{idx}.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # tutup figure biar tidak numpuk di memori
    print(f"Gambar disimpan: {output_path}")

# Create Model
print("Creating Mask R-CNN model...")
model = modellib.MaskRCNN(
    mode="training",
    config=config,
    model_dir='/content/drive/MyDrive/chicken_project/logs'
)

model.load_weights(
    filepath='mask_rcnn_coco.h5',
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"]
)

model.train(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=1,
    layers='heads'
)

model_path = 'chicken_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)

# Prediciton

# Evaluation