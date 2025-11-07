import json
import os
import sys
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from mrcnn.visualize import display_top_masks, display_instances

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

class ChickenConfig(Config):
    """Configuration for chicken detection training."""
    
    NAME = "chicken"
    
    # GPU configuration
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  # Reduce to 1 if out of memory
    
    # Number of classes (background + chicken)
    NUM_CLASSES = 1 + 1
    
    # Image dimensions
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Training parameters
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
    LEARNING_RATE = 0.001
    
    # Detection parameters
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.3
    
    # ROI parameters
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 10
    
    # Anchor scales
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

class ChickenDataset(utils.Dataset):
    """Dataset class for chicken images and annotations."""
    
    def load_chickens(self, dataset_dir, subset):
        """Load chicken dataset from COCO format annotations."""
        self.add_class("chicken", 1, "chicken")
        
        json_path = os.path.join(dataset_dir, subset, "annotations.json")
        img_dir = os.path.join(dataset_dir, subset, "images")
        
        self._validate_paths(json_path, img_dir)
        
        with open(json_path) as f:
            coco_data = json.load(f)
        
        self._add_images(coco_data, img_dir)
    
    def _validate_paths(self, json_path, img_dir):
        """Validate that required paths exist."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Annotations not found: {json_path}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    def _add_images(self, coco_data, img_dir):
        """Add images to dataset."""
        for img_info in coco_data["images"]:
            image_id = img_info["id"]
            annotations = [
                ann for ann in coco_data["annotations"] 
                if ann["image_id"] == image_id
            ]
            
            self.add_image(
                "chicken",
                image_id=image_id,
                path=os.path.join(img_dir, img_info["file_name"]),
                width=img_info["width"],
                height=img_info["height"],
                annotations=annotations
            )
    
    def load_mask(self, image_id):
        """Load mask for given image ID."""
        image_info = self.image_info[image_id]
        annotations = image_info.get("annotations", [])
        
        if not annotations:
            return np.empty((0, 0, 0)), np.array([], dtype=np.int32)
        
        masks = []
        class_ids = []
        
        for ann in annotations:
            segmentation = ann.get("segmentation")
            if segmentation:
                mask = self._create_mask_from_segmentation(
                    segmentation, 
                    image_info["height"], 
                    image_info["width"]
                )
                if mask is not None:
                    masks.append(mask)
                    class_ids.append(ann["category_id"])
        
        if masks:
            masks = np.stack(masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        
        return np.empty((0, 0, 0)), np.array([], dtype=np.int32)
    
    def _create_mask_from_segmentation(self, segmentation, height, width):
        """Create mask from segmentation data."""
        try:
            # Try to use pycocotools if available
            from pycocotools import mask as maskUtils
            rles = maskUtils.frPyObjects(segmentation, height, width)
            rle = maskUtils.merge(rles) if isinstance(rles, (list, tuple)) else rles
            return maskUtils.decode(rle)
        except ImportError:
            # Fallback: create simple rectangular mask from bbox
            bbox = segmentation[0] if isinstance(segmentation[0], list) else segmentation
            if len(bbox) >= 4:
                mask = np.zeros((height, width), dtype=np.uint8)
                x1, y1, w, h = bbox[:4]
                x2, y2 = x1 + w, y1 + h
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
                return mask
        return None

class ChickenDetectionTrainer:
    """Main class for training and evaluating chicken detection model."""
    
    def __init__(self, base_dir="/content/drive/MyDrive"):
        self.base_dir = base_dir
        self.dataset_path = os.path.join(base_dir, 'ChickenDataset')
        self.train_path = os.path.join(self.dataset_path, 'training')
        self.val_path = os.path.join(self.dataset_path, 'validation')
        
        self.config = ChickenConfig()
        self.model = None
        
    def check_dataset_structure(self):
        """Check and display dataset structure."""
        print("🔍 Checking dataset structure...")
        
        for subset, path in [("Train", self.train_path), ("Validation", self.val_path)]:
            images_dir = os.path.join(path, 'images')
            annotations_file = os.path.join(path, 'annotations.json')
            
            # Check images
            if os.path.exists(images_dir):
                image_files = [
                    f for f in os.listdir(images_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                print(f"📸 {subset} images: {len(image_files)} files")
            else:
                print(f"❌ {subset} images directory not found")
            
            # Check annotations
            if os.path.exists(annotations_file):
                with open(annotations_file) as f:
                    annotations_data = json.load(f)
                images_count = len(annotations_data.get('images', []))
                annotations_count = len(annotations_data.get('annotations', []))
                print(f"📝 {subset} annotations: {images_count} images, {annotations_count} annotations")
            else:
                print(f"❌ {subset} annotations.json not found")
    
    def create_annotations(self, image_dir, output_json):
        """Create COCO format annotations for images in directory."""
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return False
        
        images = []
        annotations = []
        categories = [{"id": 1, "name": "chicken"}]
        
        annotation_id = 1
        
        for i, image_file in enumerate(image_files):
            try:
                image_path = os.path.join(image_dir, image_file)
                with Image.open(image_path) as img:
                    width, height = img.size
                
                image_id = i + 1
                images.append({
                    "id": image_id,
                    "file_name": image_file,
                    "width": width,
                    "height": height
                })
                
                # Create centered bounding box annotation
                bbox_width, bbox_height = min(150, width//3), min(150, height//3)
                x_center, y_center = width // 2, height // 2
                
                bbox = [
                    max(0, x_center - bbox_width // 2),
                    max(0, y_center - bbox_height // 2),
                    bbox_width,
                    bbox_height
                ]
                
                segmentation = [[
                    bbox[0], bbox[1],
                    bbox[0] + bbox[2], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3],
                    bbox[0], bbox[1] + bbox[3]
                ]]
                
                area = bbox_width * bbox_height
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                
                annotation_id += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {image_file}: {e}")
                continue
        
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✅ Created annotations for {len(images)} images at: {output_json}")
        return True
    
    def prepare_datasets(self):
        """Prepare training and validation datasets."""
        # Create annotations if they don't exist
        train_annotations = os.path.join(self.train_path, 'annotations.json')
        val_annotations = os.path.join(self.val_path, 'annotations.json')
        
        if not os.path.exists(train_annotations):
            self.create_annotations(
                os.path.join(self.train_path, 'images'),
                train_annotations
            )
        
        if not os.path.exists(val_annotations):
            self.create_annotations(
                os.path.join(self.val_path, 'images'),
                val_annotations
            )
        
        # Load datasets
        dataset_train = ChickenDataset()
        dataset_train.load_chickens(self.dataset_path, "training")
        dataset_train.prepare()
        
        dataset_val = ChickenDataset()
        dataset_val.load_chickens(self.dataset_path, "validation")
        dataset_val.prepare()
        
        # Display dataset summary
        self._print_dataset_summary(dataset_train, dataset_val)
        
        return dataset_train, dataset_val
    
    def _print_dataset_summary(self, train_dataset, val_dataset):
        """Print dataset summary."""
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"Training images: {len(train_dataset.image_ids)}")
        print(f"Validation images: {len(val_dataset.image_ids)}")
        print(f"Classes: {train_dataset.class_names}")
        print("="*70)
    
    def train_model(self, dataset_train, dataset_val, epochs=1):
        """Train the Mask R-CNN model."""
        print("Creating Mask R-CNN model...")
        
        # Create model
        self.model = modellib.MaskRCNN(
            mode="training",
            config=self.config,
            model_dir=os.path.join(self.base_dir, 'chicken_project', 'logs')
        )
        
        # Load COCO weights
        coco_weights_path = os.path.join("/content/Mask_RCNN", "mask_rcnn_coco.h5")
        if not os.path.exists(coco_weights_path):
            raise FileNotFoundError(f"COCO weights not found at {coco_weights_path}")
        
        self.model.load_weights(
            filepath=coco_weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        )
        
        # Train model
        print("Starting training...")
        self.model.train(
            train_dataset=dataset_train,
            val_dataset=dataset_val,
            learning_rate=self.config.LEARNING_RATE,
            epochs=epochs,
            layers='heads'
        )
        
        # Save model
        model_path = os.path.join(self.base_dir, 'chicken_project', 'chicken_mask_rcnn_trained.h5')
        self.model.keras_model.save_weights(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        return model_path
    
    def evaluate_model(self, dataset_val, model_path, num_samples=10):
        """Evaluate the trained model."""
        # Create inference model
        inference_config = ChickenConfig()
        inference_config.GPU_COUNT = 1
        inference_config.IMAGES_PER_GPU = 1
        
        model = modellib.MaskRCNN(
            mode="inference",
            config=inference_config,
            model_dir=os.path.join(self.base_dir, 'chicken_project', 'logs')
        )
        
        model.load_weights(model_path, by_name=True)
        
        # Compute mAP
        image_ids = np.random.choice(dataset_val.image_ids, num_samples)
        APs = []
        
        for image_id in image_ids:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset_val, inference_config, image_id, use_mini_mask=False
            )
            
            results = model.detect([image], verbose=0)
            r = results[0]
            
            AP, _, _, _ = utils.compute_ap(
                gt_bbox, gt_class_id, gt_mask,
                r["rois"], r["class_ids"], r["scores"], r['masks']
            )
            APs.append(AP)
        
        mean_ap = np.mean(APs)
        print(f"mAP @ IoU=0.5: {mean_ap:.3f}")
        
        return mean_ap, model

def main():
    """Main execution function."""
    # Initialize trainer
    trainer = ChickenDetectionTrainer()
    
    # Check dataset structure
    trainer.check_dataset_structure()
    
    # Prepare datasets
    dataset_train, dataset_val = trainer.prepare_datasets()
    
    # Display sample images
    print("\nDisplaying sample training images...")
    image_ids = np.random.choice(dataset_train.image_ids, 2)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        display_top_masks(image, mask, class_ids, dataset_train.class_names)
    
    # Train model
    model_path = trainer.train_model(dataset_train, dataset_val, epochs=1)
    
    # Evaluate model
    mean_ap, inference_model = trainer.evaluate_model(dataset_val, model_path)
    
    # Test on a random image
    print("\nTesting on a random validation image...")
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_val, ChickenConfig(), image_id, use_mini_mask=False
    )
    
    results = inference_model.detect([original_image], verbose=1)
    r = results[0]
    
    display_instances(
        original_image, r['rois'], r['masks'], r['class_ids'],
        dataset_val.class_names, r['scores']
    )

if __name__ == "__main__":
    main()