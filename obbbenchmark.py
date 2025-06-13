import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import psutil
import subprocess
import threading
from collections import deque

class EdgeTPUMonitor:
    def __init__(self):
        self.inference_times = deque(maxlen=100)
        self.monitoring = False
        self.monitor_thread = None
        
    def check_edge_tpu_status(self):
        """Check if Edge TPU is detected and available"""
        print("=== Edge TPU Status Check ===")
        
        # Method 1: Check lsusb for Coral devices
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if 'Google Inc.' in result.stdout or 'Coral' in result.stdout:
                print("✓ Coral Edge TPU device detected via USB")
                for line in result.stdout.split('\n'):
                    if 'Google Inc.' in line or 'Coral' in line:
                        print(f"  Device: {line.strip()}")
            else:
                print("✗ No Coral Edge TPU device found in USB devices")
        except Exception as e:
            print(f"Could not check USB devices: {e}")
        
        # Method 2: Check for Edge TPU runtime
        try:
            import tflite_runtime.interpreter as tflite
            print("✓ TensorFlow Lite runtime available")
        except ImportError:
            print("✗ TensorFlow Lite runtime not available")
            
        try:
            from pycoral.utils import edgetpu
            print("✓ PyCoral Edge TPU library available")
            
            # Try to list Edge TPU devices
            devices = edgetpu.list_edge_tpus()
            if devices:
                print(f"✓ Found {len(devices)} Edge TPU device(s):")
                for i, device in enumerate(devices):
                    print(f"  Device {i}: {device}")
            else:
                print("✗ No Edge TPU devices found by PyCoral")
                
        except ImportError:
            print("✗ PyCoral library not available")
        except Exception as e:
            print(f"Error checking Edge TPU devices: {e}")
            
    def monitor_system_resources(self):
        """Monitor CPU usage and temperature during inference"""
        print("\n=== System Resource Monitoring ===")
        print("CPU Usage | Temperature | Inference Time (ms)")
        print("-" * 50)
        
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Try to get temperature (Raspberry Pi specific)
            temp = "N/A"
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_raw = int(f.read().strip())
                    temp = f"{temp_raw / 1000:.1f}°C"
            except:
                pass
                
            if self.inference_times:
                avg_inference = np.mean(list(self.inference_times))
                print(f"{cpu_percent:8.1f}% | {temp:11s} | {avg_inference:13.1f}ms", end='\r')
            
            time.sleep(1)
    
    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_system_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("\nMonitoring stopped.")

class YOLOv8OBBDetectorWithMonitoring:
    def __init__(self, model_path='yolov8n-obb.pt'):
        self.model_path = model_path
        self.is_edgetpu = self._is_edgetpu_model(model_path)
        self.monitor = EdgeTPUMonitor()
        
        # Check Edge TPU status before loading model
        self.monitor.check_edge_tpu_status()
        
        print(f"\n=== Loading Model ===")
        if self.is_edgetpu:
            print(f"Loading Edge TPU model: {model_path}")
            self.model = YOLO(model_path, task='obb')
        else:
            print(f"Loading regular model: {model_path}")
            self.model = YOLO(model_path)

        self.colors = self._generate_colors()
        
        # Test inference speed
        self._benchmark_inference()

    def _is_edgetpu_model(self, model_path):
        return model_path.endswith('_edgetpu.tflite') or 'edgetpu' in model_path.lower()

    def _generate_colors(self):
        np.random.seed(42)
        colors = []
        for i in range(100):
            colors.append([
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            ])
        return colors
    
    def _benchmark_inference(self):
        """Benchmark inference speed to verify Edge TPU acceleration"""
        print("\n=== Inference Speed Benchmark ===")
        
        # Create a test image
        test_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        # Warmup runs
        print("Warming up...")
        for _ in range(5):
            _ = self.model(test_img, imgsz=320, verbose=False)
        
        # Benchmark runs
        times = []
        print("Running benchmark...")
        for i in range(20):
            start_time = time.time()
            results = self.model(test_img, imgsz=320, verbose=False)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(inference_time)
            print(f"Run {i+1:2d}: {inference_time:6.1f}ms")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\nBenchmark Results:")
        print(f"Average inference time: {avg_time:.1f} ± {std_time:.1f}ms")
        print(f"Min/Max: {min_time:.1f}ms / {max_time:.1f}ms")
        
        # Expected times for comparison
        print(f"\nExpected inference times:")
        print(f"Edge TPU (Coral):  5-15ms")
        print(f"CPU (Raspberry Pi): 100-500ms")
        print(f"GPU (if available): 10-50ms")
        
        if avg_time < 30:
            print("✓ Fast inference detected - likely running on Edge TPU!")
        elif avg_time > 80:
            print("⚠ Slow inference detected - might be running on CPU")
        else:
            print("? Moderate inference time - check other indicators")

    def detect_image_with_timing(self, image_path, conf_threshold=0.25, save_path=None):
        """Detect objects with detailed timing information"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        original_img = img.copy()
        img_resized = cv2.resize(img, (320, 320))

        # Detailed timing
        total_start = time.time()
        
        # Model inference timing
        inference_start = time.time()
        results = self.model(img_resized, conf=conf_threshold, imgsz=320, verbose=False)
        inference_end = time.time()
        
        # Post-processing timing
        postprocess_start = time.time()
        
        for result in results:
            if result.obb is not None:
                boxes = result.obb.xyxyxyxy.cpu().numpy()
                confidences = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy().astype(int)

                scale_x = original_img.shape[1] / 320
                scale_y = original_img.shape[0] / 320
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    scaled_box = box.copy()
                    scaled_box[::2] *= scale_x
                    scaled_box[1::2] *= scale_y
                    
                    label = self.model.names[cls]
                    color = self.colors[cls % len(self.colors)]
                    self.draw_obb(original_img, scaled_box, label, conf, color)

                print(f"Detected {len(boxes)} objects")
            else:
                print("No objects detected")
        
        postprocess_end = time.time()
        total_end = time.time()
        
        # Timing results
        inference_time = (inference_end - inference_start) * 1000
        postprocess_time = (postprocess_end - postprocess_start) * 1000
        total_time = (total_end - total_start) * 1000
        
        print(f"\nTiming breakdown:")
        print(f"Inference:      {inference_time:6.1f}ms")
        print(f"Post-processing: {postprocess_time:6.1f}ms")
        print(f"Total:          {total_time:6.1f}ms")
        
        # Store inference time for monitoring
        self.monitor.inference_times.append(inference_time)

        if save_path:
            cv2.imwrite(save_path, original_img)
            print(f"Result saved to {save_path}")

        return original_img

    def draw_obb(self, img, box, label, confidence, color):
        """Draw oriented bounding box on image"""
        points = np.array(box).reshape(4, 2).astype(int)
        cv2.polylines(img, [points], True, color, 2)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        
        label_pos = tuple(points[0])
        text = f'{label}: {confidence:.2f}'
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(img,
                     (label_pos[0], label_pos[1] - text_height - baseline),
                     (label_pos[0] + text_width, label_pos[1]),
                     color, -1)
        
        cv2.putText(img, text, label_pos, font, font_scale, (255, 255, 255), thickness)

    def detect_camera_with_monitoring(self, camera_id=0, conf_threshold=0.25):
        """Real-time detection with system monitoring"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("Starting camera detection with monitoring...")
        print("Press 'q' to quit, 's' to save frame, 'm' to toggle monitoring")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        frame_count = 0
        fps_counter = deque(maxlen=30)

        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy()
            resized_frame = cv2.resize(frame, (320, 320))

            # Timed inference
            inference_start = time.time()
            results = self.model(resized_frame, conf=conf_threshold, imgsz=320, verbose=False)
            inference_end = time.time()
            
            inference_time = (inference_end - inference_start) * 1000
            self.monitor.inference_times.append(inference_time)

            # Process results
            detection_count = 0
            for result in results:
                if result.obb is not None:
                    boxes = result.obb.xyxyxyxy.cpu().numpy()
                    confidences = result.obb.conf.cpu().numpy()
                    classes = result.obb.cls.cpu().numpy().astype(int)
                    detection_count = len(boxes)

                    scale_x = original_frame.shape[1] / 320
                    scale_y = original_frame.shape[0] / 320

                    for box, conf, cls in zip(boxes, confidences, classes):
                        scaled_box = box.copy()
                        scaled_box[::2] *= scale_x
                        scaled_box[1::2] *= scale_y
                        
                        label = self.model.names[cls]
                        color = self.colors[cls % len(self.colors)]
                        self.draw_obb(original_frame, scaled_box, label, conf, color)

            # Calculate FPS
            frame_end = time.time()
            frame_time = frame_end - frame_start
            fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
            current_fps = np.mean(fps_counter)

            # Add info text
            cv2.putText(original_frame, f'Detections: {detection_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(original_frame, f'FPS: {current_fps:.1f}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(original_frame, f'Inference: {inference_time:.1f}ms', (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('YOLOv8-OBB Detection with Monitoring', original_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_name = f'detection_frame_{frame_count}.jpg'
                cv2.imwrite(save_name, original_frame)
                print(f"Frame saved as {save_name}")
            elif key == ord('m'):
                if self.monitor.monitoring:
                    self.monitor.stop_monitoring()
                else:
                    self.monitor.start_monitoring()

            frame_count += 1

        self.monitor.stop_monitoring()
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function with comprehensive Edge TPU verification"""
    model_path = '/home/Voldemort/coral/reef/newnewmodels/best_full_integer_quant_edgetpu.tflite'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        detector = YOLOv8OBBDetectorWithMonitoring(model_path)
        print("Model loaded successfully!")
        
        # Test with image
        image_path = '/home/Voldemort/coral/reef/test/images/Screenshot-2025-01-04-161008_png.rf.09742509c0a45467c178fc4280ad7dd6.jpg'
        
        if os.path.exists(image_path):
            print("\n=== Testing with image ===")
            result_img = detector.detect_image_with_timing(image_path, save_path='monitored_output.jpg')
            
            if result_img is not None:
                cv2.imshow('Detection Result', result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        # Camera detection with monitoring
        print("\n=== Starting camera detection with monitoring ===")
        detector.detect_camera_with_monitoring()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
