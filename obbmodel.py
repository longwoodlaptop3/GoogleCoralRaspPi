import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

class YOLOv8OBBDetector:
    def __init__(self, model_path='yolov8n-obb.pt'):
        """
        Initialize YOLOv8-OBB detector
        Args:
            model_path: Path to YOLOv8-OBB model weights (.pt, .tflite, or _edgetpu.tflite)
        """
        self.model_path = model_path
        self.is_edgetpu = self._is_edgetpu_model(model_path)

        if self.is_edgetpu:
            print(f"Loading Edge TPU model: {model_path}")
            # For Edge TPU models, we need to specify the task explicitly
            self.model = YOLO(model_path, task='obb')
        else:
            self.model = YOLO(model_path)

        self.colors = self._generate_colors()

    def _is_edgetpu_model(self, model_path):
        """Check if the model is an Edge TPU model"""
        return model_path.endswith('_edgetpu.tflite') or 'edgetpu' in model_path.lower()

    def _generate_colors(self):
        """Generate random colors for different classes"""
        np.random.seed(42)
        colors = []
        for i in range(100):  # Generate 100 colors
            colors.append([
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ])
        return colors

    def draw_obb(self, img, box, label, confidence, color):
        """
        Draw oriented bounding box on image
        Args:
            img: Image to draw on
            box: OBB coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
            label: Class label
            confidence: Detection confidence
            color: Box color
        """
        # Convert box to integer points
        points = np.array(box).reshape(4, 2).astype(int)

        # Draw the oriented bounding box
        cv2.polylines(img, [points], True, color, 2)

        # Fill the box with semi-transparent color
        overlay = img.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

        # Calculate label position (top-left corner of OBB)
        label_pos = tuple(points[0])

        # Prepare label text
        text = f'{label}: {confidence:.2f}'

        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw label background
        cv2.rectangle(img,
                     (label_pos[0], label_pos[1] - text_height - baseline),
                     (label_pos[0] + text_width, label_pos[1]),
                     color, -1)

        # Draw label text
        cv2.putText(img, text, label_pos, font, font_scale, (255, 255, 255), thickness)

    def detect_image(self, image_path, conf_threshold=0.25, save_path=None):
        """
        Detect objects in a single image
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_path: Path to save output image (optional)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Store original image for drawing results
        original_img = img.copy()
        
        # Resize image to match model input size (320x320 for your Edge TPU model)
        img_resized = cv2.resize(img, (320, 320))

        # Run inference with imgsz parameter to ensure correct input size
        results = self.model(img_resized, conf=conf_threshold, imgsz=320)

        # Process results
        for result in results:
            if result.obb is not None:
                boxes = result.obb.xyxyxyxy.cpu().numpy()  # OBB coordinates
                confidences = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy().astype(int)

                # Scale boxes back to original image size
                scale_x = original_img.shape[1] / 320
                scale_y = original_img.shape[0] / 320
                
                # Draw each detection on original image
                for box, conf, cls in zip(boxes, confidences, classes):
                    # Scale box coordinates back to original image size
                    scaled_box = box.copy()
                    scaled_box[::2] *= scale_x  # Scale x coordinates
                    scaled_box[1::2] *= scale_y  # Scale y coordinates
                    
                    label = self.model.names[cls]
                    color = self.colors[cls % len(self.colors)]
                    self.draw_obb(original_img, scaled_box, label, conf, color)

                print(f"Detected {len(boxes)} objects")
            else:
                print("No objects detected")

        # Save or display result
        if save_path:
            cv2.imwrite(save_path, original_img)
            print(f"Result saved to {save_path}")

        return original_img

    def detect_camera(self, camera_id=0, conf_threshold=0.25):
        """
        Real-time detection using USB camera
        Args:
            camera_id: Camera ID (usually 0 for default camera)
            conf_threshold: Confidence threshold for detections
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("Press 'q' to quit, 's' to save current frame")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Store original frame for drawing
            original_frame = frame.copy()
            
            # Resize frame to model input size
            resized_frame = cv2.resize(frame, (320, 320))

            # Run inference with correct image size
            results = self.model(resized_frame, conf=conf_threshold, imgsz=320)

            # Process results
            detection_count = 0
            for result in results:
                if result.obb is not None:
                    boxes = result.obb.xyxyxyxy.cpu().numpy()
                    confidences = result.obb.conf.cpu().numpy()
                    classes = result.obb.cls.cpu().numpy().astype(int)

                    detection_count = len(boxes)

                    # Scale boxes back to original frame size
                    scale_x = original_frame.shape[1] / 320
                    scale_y = original_frame.shape[0] / 320

                    # Draw each detection on original frame
                    for box, conf, cls in zip(boxes, confidences, classes):
                        # Scale box coordinates back to original frame size
                        scaled_box = box.copy()
                        scaled_box[::2] *= scale_x  # Scale x coordinates
                        scaled_box[1::2] *= scale_y  # Scale y coordinates
                        
                        label = self.model.names[cls]
                        color = self.colors[cls % len(self.colors)]
                        self.draw_obb(original_frame, scaled_box, label, conf, color)

            # Add info text
            cv2.putText(original_frame, f'Detections: {detection_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(original_frame, f'Frame: {frame_count}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('YOLOv8-OBB Detection', original_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_name = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(save_name, original_frame)
                print(f"Frame saved as {save_name}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function with usage examples"""
    # Initialize detector with Edge TPU model
    model_path = '/home/Voldemort/coral/reef/models/best_full_integer_quant_edgetpu.tflite'  # Your Edge TPU model path

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please check the model path and ensure the file exists.")
        return

    try:
        detector = YOLOv8OBBDetector(model_path)
        print("Edge TPU model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting steps:")
        print("1. Install Edge TPU runtime: sudo apt install libedgetpu1-std")
        print("2. Install required packages: pip install pycoral tflite-runtime")
        print("3. Ensure your model is properly exported for Edge TPU")
        return

    print("\n=== Camera Detection ===")
    print("Starting camera detection...")
    detector.detect_camera(camera_id=0, conf_threshold=0.25)

if __name__ == "__main__":
    main()
