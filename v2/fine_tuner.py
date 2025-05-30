import os
from ultralytics import YOLO
import yaml
import shutil
from pathlib import Path

class YOLO11FineTuner:
    def __init__(self, base_model='yolo11s.pt'):
        """
        Initialize YOLO11 fine-tuner

        Args:
            base_model: Pre-trained YOLO11 model to start from
                       - yolo11n.pt (nano - fastest)
                       - yolo11s.pt (small - recommended)
                       - yolo11m.pt (medium - better accuracy)
        """
        self.base_model = base_model
        self.model = None

    def prepare_dataset(self, roboflow_dataset_path, output_path='./classroom_dataset'):
        """
        Prepare Roboflow dataset for YOLO11 training

        Args:
            roboflow_dataset_path: Path to downloaded Roboflow dataset
            output_path: Where to organize the final dataset
        """
        print("Preparing dataset...")

        # Create output directory structure
        os.makedirs(output_path, exist_ok=True)

        # Copy the Roboflow dataset
        if os.path.exists(roboflow_dataset_path):
            shutil.copytree(roboflow_dataset_path, output_path, dirs_exist_ok=True)
            print(f"Dataset copied to {output_path}")
        else:
            print(f"Dataset path {roboflow_dataset_path} not found!")
            return False

        # Verify dataset structure
        required_files = ['data.yaml', 'train', 'valid']
        for item in required_files:
            if not os.path.exists(os.path.join(output_path, item)):
                print(f"Missing required file/folder: {item}")
                return False

        print("Dataset structure verified ✓")
        return True

    def modify_yaml_config(self, dataset_path):
        """
        Modify the data.yaml file to include new classes alongside COCO classes

        Args:
            dataset_path: Path to dataset containing data.yaml
            new_classes: List of new class names to add
        """
        yaml_path = os.path.join(dataset_path, 'data.yaml')

        # Read existing YAML
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        print(f"Adding New classes: {data.get('names', [])}")
        new_classes = data.get('names', [])

        # COCO classes (first 80 classes)
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Combine COCO classes with new classes
        all_classes = coco_classes + new_classes

        # Update YAML configuration
        data['nc'] = len(all_classes)  # Number of classes
        data['names'] = all_classes

        # Write updated YAML
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print(f"Updated classes ({len(all_classes)} total): {all_classes}")
        print(f"New classes added: {new_classes}")

    def start_training(self, dataset_path, epochs=100, imgsz=640, batch_size=16):
        """
        Start fine-tuning the YOLO11 model

        Args:
            dataset_path: Path to prepared dataset
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size (adjust based on GPU memory)
        """
        print(f"Starting fine-tuning with {self.base_model}...")

        # Load pre-trained model
        self.model = YOLO(self.base_model)

        # Start training
        results = self.model.train(
            data=os.path.join(dataset_path, 'data.yaml'),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='classroom_finetuning',
            patience=10,  # Early stopping patience
            save=True,
            plots=True,
            verbose=True
        )

        print("Training completed!")
        return results

    def validate_model(self, dataset_path):
        """
        Validate the fine-tuned model

        Args:
            dataset_path: Path to dataset for validation
        """
        if self.model is None:
            print("No model loaded. Train first or load a trained model.")
            return

        print("Validating model...")
        results = self.model.val(data=os.path.join(dataset_path, 'data.yaml'))
        return results

    def test_detection(self, image_path, conf_threshold=0.25):
        """
        Test the fine-tuned model on an image

        Args:
            image_path: Path to test image
            conf_threshold: Confidence threshold for detections
        """
        if self.model is None:
            # Load the best trained model
            self.model = YOLO('runs/detect/classroom_finetuning/weights/best.pt')

        # Run detection
        results = self.model(image_path, conf=conf_threshold)

        # Display results
        for result in results:
            result.show()

        return results

def main():
    """
    Main function to run YOLO11 fine-tuning
    """
    print("YOLO11 Fine-tuning for Classroom Objects")
    print("=" * 50)

    # Step 1: Initialize fine-tuner
    fine_tuner = YOLO11FineTuner(base_model='yolo11s.pt')

    # Step 2: Prepare dataset (replace with your Roboflow dataset path)
    roboflow_path = input("Enter path to your Roboflow dataset: ").strip()
    if not roboflow_path:
        roboflow_path = "./roboflow_dataset"  # Default path

    dataset_prepared = fine_tuner.prepare_dataset(
        roboflow_dataset_path=roboflow_path,
        output_path='./classroom_dataset'
    )

    if not dataset_prepared:
        print("Dataset preparation failed!")
        return

    # Step 3: Modify YAML configuration
    fine_tuner.modify_yaml_config('./classroom_dataset')

    # Step 4: Start training
    print("\nStarting training...")
    print("Recommended settings:")
    print("- epochs=50-100 (start with 50)")
    print("- batch_size=8-16 (depends on GPU memory)")
    print("- imgsz=640 (standard)")

    epochs = int(input("Enter number of epochs (default 50): ") or "50")
    batch_size = int(input("Enter batch size (default 8): ") or "8")

    results = fine_tuner.start_training(
        dataset_path='./classroom_dataset',
        epochs=epochs,
        batch_size=batch_size
    )

    # Step 5: Validate model
    print("\nValidating model...")
    fine_tuner.validate_model('./classroom_dataset')

    # Step 6: Test on sample image
    test_image = input("Enter path to test image (optional): ").strip()
    if test_image and os.path.exists(test_image):
        print("Testing model...")
        fine_tuner.test_detection(test_image)

    print("\nFine-tuning completed!")
    print("Your model is saved in: runs/detect/classroom_finetuning/weights/best.pt")

if __name__ == "__main__":
    main()
