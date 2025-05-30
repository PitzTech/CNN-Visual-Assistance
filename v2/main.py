import cv2
import numpy as np
from ultralytics import YOLO
import time

class ClassroomObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize the classroom object detector

        Args:
            model_path: Path to YOLO model (will download if not exists)
            confidence_threshold: Minimum confidence for detections
        """
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Define classroom-related classes from COCO dataset
        self.classroom_classes = {
            0: 'person',
            56: 'chair',
            60: 'dining table',  # can represent classroom tables
            62: 'laptop',
            63: 'mouse',
            64: 'remote',
            65: 'keyboard',
            66: 'cell phone',
            67: 'microwave',  # might detect some electronics
            72: 'tv',
            73: 'laptop',
            74: 'mouse',
            75: 'remote',
            76: 'keyboard',
            77: 'cell phone',
            84: 'book',
            85: 'clock',
            86: 'vase',  # decorative items
            87: 'scissors',
            88: 'teddy bear',
            89: 'hair drier',
            90: 'toothbrush'
        }

        # Colors for bounding boxes (BGR format)
        self.colors = {
            'person': (255, 144, 30),
            'chair': (255, 178, 50),
            'dining table': (0, 255, 0),
            'laptop': (255, 0, 255),
            'book': (0, 255, 255),
            'cell phone': (255, 255, 0),
            'clock': (128, 0, 128),
            'keyboard': (255, 165, 0),
            'mouse': (0, 128, 255),
            'scissors': (255, 20, 147),
            'default': (0, 255, 0)
        }

    def detect_objects(self, frame):
        """
        Detect objects in a frame

        Args:
            frame: Input frame from webcam

        Returns:
            frame: Frame with bounding boxes and labels
            detections: List of detected objects
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Get class name
                    class_name = self.model.names[class_id]

                    # Store detection
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2)
                    }
                    detections.append(detection)

                    # Draw bounding box
                    color = self.colors.get(class_name, self.colors['default'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

                    # Background rectangle for text
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)

                    # Text
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame, detections

    def add_info_panel(self, frame, detections, fps):
        """
        Add information panel to the frame

        Args:
            frame: Input frame
            detections: List of detected objects
            fps: Current FPS

        Returns:
            frame: Frame with info panel
        """
        height, width = frame.shape[:2]

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Add title
        cv2.putText(frame, "Classroom Object Detection", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Count objects
        object_counts = {}
        for detection in detections:
            class_name = detection['class']
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        # Display object counts
        y_offset = 80
        cv2.putText(frame, f"Detected Objects: {len(detections)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 20
        for obj_class, count in list(object_counts.items())[:3]:  # Show top 3
            cv2.putText(frame, f"{obj_class}: {count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15

        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def run_detection(self, camera_id=0):
        """
        Run real-time object detection

        Args:
            camera_id: Camera device ID (default: 0)
        """
        print(f"Starting webcam (Camera ID: {camera_id})...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # FPS calculation
        fps_counter = 0
        fps_timer = time.time()
        fps = 0

        print("Detection started! Press 'q' to quit, 's' to save screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect objects
            frame, detections = self.detect_objects(frame)

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            # Add info panel
            frame = self.add_info_panel(frame, detections, fps)

            # Display frame
            cv2.imshow('Classroom Object Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"classroom_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

def main():
    """
    Main function to run the classroom object detector
    """
    print("Classroom Object Detection System")
    print("=" * 40)

    try:
        # Initialize detector
        detector = ClassroomObjectDetector(
            model_path='yolov8n.pt',  # You can change to yolov8s.pt, yolov8m.pt, etc. for better accuracy
            confidence_threshold=0.5
        )

        # Start detection
        detector.run_detection(camera_id=0)

    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required packages installed:")
        print("pip install ultralytics opencv-python")

if __name__ == "__main__":
    main()
