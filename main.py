import json
import os
import sys
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from mrcnn import model as modellib, utils
from mrcnn.config import Config

# Add Mask_RCNN to path
sys.path.append('/content/Mask_RCNN')

try:
    from mrcnn.visualize import display_top_masks, display_instances
except ImportError:
    print("⚠️ Visualization functions not available, continuing without them...")

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# Fix for TensorFlow 1.15.5 and Keras 2.3.1 compatibility
import keras
print(f"Keras version: {keras.__version__}")

class ChickenConfig(Config):
    """Configuration for chicken detection training."""
    
    NAME = "chicken"
    
    # GPU configuration
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Reduced for Colab memory constraints
    
    # Number of classes (background + chicken)
    NUM_CLASSES = 1 + 1
    
    # Image dimensions
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Training parameters
    STEPS_PER_EPOCH = 50  # Reduced for demo
    VALIDATION_STEPS = 10
    LEARNING_RATE = 0.001
    
    # Detection parameters
    DETECTION_MIN_CONFIDENCE = 0.8  # Lowered for better detection
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
        # Create mapping from image_id to annotations
        annotations_by_image = {}
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        for img_info in coco_data["images"]:
            image_id = img_info["id"]
            annotations = annotations_by_image.get(image_id, [])
            
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
            return np.empty([image_info["height"], image_info["width"], 0]), np.array([], dtype=np.int32)
        
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
                    # Ensure mask is 2D
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    masks.append(mask)
                    class_ids.append(ann["category_id"])
        
        if masks:
            masks = np.stack(masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        
        return np.empty([image_info["height"], image_info["width"], 0]), np.array([], dtype=np.int32)
    
    def _create_mask_from_segmentation(self, segmentation, height, width):
        """Create mask from segmentation data."""
        try:
            # Try to use pycocotools if available
            from pycocotools import mask as maskUtils
            
            if isinstance(segmentation, list):
                if isinstance(segmentation[0], list):
                    # Polygon format
                    rles = maskUtils.frPyObjects(segmentation, height, width)
                    rle = maskUtils.merge(rles)
                else:
                    # BBox format [x,y,width,height]
                    if len(segmentation) == 4:
                        x, y, w, h = segmentation
                        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                        rles = maskUtils.frPyObjects(segmentation, height, width)
                        rle = maskUtils.merge(rles)
                    else:
                        # Assume it's already RLE
                        rle = segmentation
            else:
                # Assume it's RLE
                rle = segmentation
                
            mask = maskUtils.decode(rle)
            return mask
            
        except ImportError:
            print("⚠️ pycocotools not available, using fallback mask creation")
            # Fallback: create simple rectangular mask from bbox
            if isinstance(segmentation, list) and len(segmentation) >= 4:
                bbox = segmentation
                if isinstance(bbox[0], list):
                    bbox = bbox[0]  # Take first polygon
                
                if len(bbox) >= 4:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    if len(bbox) == 4:
                        # [x, y, width, height] format
                        x, y, w, h = bbox
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    else:
                        # Polygon format [x1, y1, x2, y2, ...]
                        x_coords = bbox[0::2]
                        y_coords = bbox[1::2]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                    
                    mask[int(y1):int(y2), int(x1):int(x2)] = 1
                    return mask
        
        return None


class FixedMaskRCNN(modellib.MaskRCNN):
    """Fixed version of MaskRCNN to handle compatibility issues."""
    
    def compile(self, learning_rate, momentum):
        """Compile the model with fixes for TensorFlow 1.15.5 compatibility."""
        from keras.optimizers import SGD
        from keras import losses
        
        # Model compilation
        optimizer = SGD(lr=learning_rate, momentum=momentum)
        
        # Add losses directly without metrics_tensors
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"
        ]
        
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (tf.reduce_mean(layer.output) * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)
        
        # Add custom metrics without using metrics_tensors
        self.keras_model.metrics_names = []
        self.keras_model.metrics_tensors = []  # This will be created if needed
        
        # Regular compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs)
        )
        
        # Add custom metrics manually
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            # Use older Keras method for metrics
            if hasattr(self.keras_model, 'metrics_tensors'):
                self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output))


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
                print(f"   Expected at: {annotations_file}")
    
    def create_annotations(self, image_dir, output_json):
        """Create COCO format annotations for images in directory."""
        if not os.path.exists(image_dir):
            print(f"❌ Image directory not found: {image_dir}")
            return False
            
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
                
                # Create centered bounding box annotation (covering 30% of image)
                bbox_width = int(width * 0.3)
                bbox_height = int(height * 0.3)
                x_center, y_center = width // 2, height // 2
                
                bbox = [
                    max(0, x_center - bbox_width // 2),
                    max(0, y_center - bbox_height // 2),
                    bbox_width,
                    bbox_height
                ]
                
                # Create polygon (rectangle) from bbox
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
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✅ Created annotations for {len(images)} images at: {output_json}")
        return True
    
    def prepare_datasets(self):
        """Prepare training and validation datasets."""
        print("\n📁 Preparing datasets...")
        
        # Create annotations if they don't exist
        train_annotations = os.path.join(self.train_path, 'annotations.json')
        val_annotations = os.path.join(self.val_path, 'annotations.json')
        
        if not os.path.exists(train_annotations):
            print("Creating training annotations...")
            self.create_annotations(
                os.path.join(self.train_path, 'images'),
                train_annotations
            )
        
        if not os.path.exists(val_annotations):
            print("Creating validation annotations...")
            self.create_annotations(
                os.path.join(self.val_path, 'images'),
                val_annotations
            )
        
        # Load datasets
        print("Loading training dataset...")
        dataset_train = ChickenDataset()
        dataset_train.load_chickens(self.dataset_path, "training")
        dataset_train.prepare()
        
        print("Loading validation dataset...")
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
        print("\n🎯 Starting model training...")
        
        # Create model directory
        model_dir = os.path.join(self.base_dir, 'chicken_project', 'logs')
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model with fixed version
        print("Creating FixedMaskRCNN model...")
        self.model = FixedMaskRCNN(
            mode="training",
            config=self.config,
            model_dir=model_dir
        )
        
        # Load COCO weights
        coco_weights_path = "/content/Mask_RCNN/mask_rcnn_coco.h5"
        if not os.path.exists(coco_weights_path):
            print(f"❌ COCO weights not found at {coco_weights_path}")
            return None
        
        print("Loading COCO weights...")
        self.model.load_weights(
            filepath=coco_weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        )
        
        # Train model
        print(f"Training for {epochs} epochs...")
        
        # Use custom training loop to avoid compatibility issues
        self._custom_train(dataset_train, dataset_val, epochs)
        
        # Save model
        model_path = os.path.join(self.base_dir, 'chicken_project', 'chicken_mask_rcnn_trained.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.keras_model.save_weights(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        return model_path
    
    def _custom_train(self, train_dataset, val_dataset, epochs):
        """Custom training loop to avoid compatibility issues."""
        from keras.engine.training import GeneratorEnqueuer
        
        # Simple training without advanced features
        self.model.set_trainable(True)
        self.model.keras_model._make_train_function()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Simple training loop
            train_generator = self.model.data_generator(
                train_dataset, self.config, shuffle=True,
                augmentation=None,
                batch_size=self.config.BATCH_SIZE
            )
            
            # Train for one epoch
            steps_per_epoch = min(self.config.STEPS_PER_EPOCH, len(train_dataset.image_ids))
            
            for step in range(steps_per_epoch):
                batch = next(train_generator)
                losses = self.model.keras_model.train_on_batch(batch[:4], batch[4])
                
                if step % 10 == 0:
                    print(f"Step {step}/{steps_per_epoch} - Loss: {losses}")
    
    def evaluate_model(self, dataset_val, model_path, num_samples=5):
        """Evaluate the trained model."""
        print(f"\n📊 Evaluating model on {num_samples} samples...")
        
        # Create inference config
        inference_config = ChickenConfig()
        inference_config.GPU_COUNT = 1
        inference_config.IMAGES_PER_GPU = 1
        inference_config.BATCH_SIZE = 1
        
        # Create inference model
        model = modellib.MaskRCNN(
            mode="inference",
            config=inference_config,
            model_dir=os.path.join(self.base_dir, 'chicken_project', 'logs')
        )
        
        model.load_weights(model_path, by_name=True)
        
        # Compute mAP
        available_samples = min(num_samples, len(dataset_val.image_ids))
        image_ids = np.random.choice(dataset_val.image_ids, available_samples, replace=False)
        APs = []
        
        for i, image_id in enumerate(image_ids):
            try:
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
                print(f"Sample {i+1}/{available_samples}: AP = {AP:.3f}")
                
            except Exception as e:
                print(f"⚠️ Error evaluating sample {i+1}: {e}")
                continue
        
        if APs:
            mean_ap = np.mean(APs)
            print(f"🎯 mAP @ IoU=0.5: {mean_ap:.3f}")
        else:
            mean_ap = 0.0
            print("❌ Could not compute mAP")
        
        return mean_ap, model


def main():
    """Main execution function."""
    print("🐔 Chicken Detection Training Started!")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = ChickenDetectionTrainer()
        
        # Check dataset structure
        trainer.check_dataset_structure()
        
        # Prepare datasets
        dataset_train, dataset_val = trainer.prepare_datasets()
        
        # Display sample images if visualization is available
        try:
            if len(dataset_train.image_ids) > 0:
                print("\n🖼️ Displaying sample training images...")
                image_ids = np.random.choice(dataset_train.image_ids, min(2, len(dataset_train.image_ids)), replace=False)
                for image_id in image_ids:
                    image = dataset_train.load_image(image_id)
                    mask, class_ids = dataset_train.load_mask(image_id)
                    display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=1)
        except Exception as e:
            print(f"⚠️ Could not display sample images: {e}")
        
        # Train model
        model_path = trainer.train_model(dataset_train, dataset_val, epochs=1)
        
        if model_path and os.path.exists(model_path):
            # Evaluate model
            mean_ap, inference_model = trainer.evaluate_model(dataset_val, model_path, num_samples=5)
            
            # Test on a random image
            if len(dataset_val.image_ids) > 0:
                print("\n🧪 Testing on a random validation image...")
                try:
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
                    
                except Exception as e:
                    print(f"⚠️ Could not display test results: {e}")
        
        print("\n✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)