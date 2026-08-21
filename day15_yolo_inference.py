from ultralytics import YOLO
import numpy as np

def run_yolo_inference():
    model = YOLO('yolov8n.pt')
    
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("=== YOLOv8 Inference Start ===")
    results = model(dummy_frame, verbose=False)
    
    # 예측된 박스 정보 추출
    boxes = results[0].boxes
    
    print(f"Input frame shape: {dummy_frame.shape}")
    print(f"Detected boxes: {len(boxes)}") 

if __name__ == "__main__":
    run_yolo_inference()