import os
import json
import numpy as np
import sys
import time
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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

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

# Fungsi untuk log tahapan
def log_step(title):
    print("\n" + "=" * 70)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {title}")
    print("=" * 70)

# Fungsi untuk log sub-progress
def log_info(msg):
    print(f"  → {msg}")

# === 1️⃣ Setup Awal ===
log_step("INISIALISASI KONFIGURASI DAN PATH")

config = ChickenConfig()

ROOT_DIR = os.path.abspath("/content/Mask_RCNN/mrcnn")
sys.path.append(ROOT_DIR)

COCO_WEIGHTS_PATH = os.path.join("/content/Mask_RCNN", "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join("/content/Mask_RCNN", "logs")

log_info(f"Root Dir        : {ROOT_DIR}")
log_info(f"COCO Weights    : {COCO_WEIGHTS_PATH}")
log_info(f"Logs Directory  : {DEFAULT_LOGS_DIR}")

# === 2️⃣ Load Dataset ===
log_step("MEMUAT DATASET LATIHAN DAN VALIDASI")

train_dir = '/content/drive/MyDrive/ChickenDataset'
val_dir   = '/content/drive/MyDrive/ChickenDataset'

dataset_train = ChickenDataset()
dataset_train.load_chickens(train_dir, "training")
dataset_train.prepare()
log_info(f"Dataset training dimuat dari: {train_dir}")

dataset_val = ChickenDataset()
dataset_val.load_chickens(val_dir, "validation")
dataset_val.prepare()
log_info(f"Dataset validasi dimuat dari: {val_dir}")

# === 3️⃣ Summary Dataset ===
log_step("RINGKASAN DATASET")
print(f"Training images : {len(dataset_train.image_ids)}")
print(f"Validation imgs : {len(dataset_val.image_ids)}")
print(f"Classes         : {dataset_train.class_names}")

# === 4️⃣ Visualisasi Sampel ===
log_step("GENERATE SAMPEL VISUALISASI DAN SIMPAN KE GOOGLE DRIVE")

# Path tujuan di Google Drive
drive_sample_dir = '/content/drive/MyDrive/ChickenDataset/samples'
if not os.path.exists(drive_sample_dir):
    os.makedirs(drive_sample_dir, exist_ok=True)
    log_info(f"Folder belum ada. Membuat folder baru di Google Drive: {drive_sample_dir}")
else:
    log_info(f"Folder sudah ada: {drive_sample_dir}")

# Pilih acak 1 gambar untuk divisualisasikan
image_ids = np.random.choice(dataset_train.image_ids, 1)

for idx, image_id in enumerate(image_ids):
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Simpan hasil visualisasi ke Google Drive
    output_path = os.path.join(drive_sample_dir, f"output_{idx}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    log_info(f"Gambar disimpan di Google Drive: {output_path}")


model = modellib.MaskRCNN(
    mode="training",
    config=config,
    model_dir='/content/drive/MyDrive/chicken_project/logs'
)
log_info("Model Mask R-CNN berhasil dibuat.")

# === 6️⃣ Load Pretrained Weights ===
log_step("MEMUAT BOBOT COCO (TRANSFER LEARNING)")

model.load_weights(
    filepath='mask_rcnn_coco.h5',
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)
log_info("Bobot COCO berhasil dimuat.")

# === 7️⃣ Proses Training ===
log_step("MEMULAI PROSES TRAINING")

model.train(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=1,
    layers='heads'
)

# === 8️⃣ Simpan Model Hasil Training ===
log_step("MENYIMPAN HASIL MODEL TERLATIH")

model_path = 'chicken_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
log_info(f"Model tersimpan: {model_path}")

# Prediciton

# Evaluation