def calculate_iou(box1, box2):
    # 어제 만든 IoU 함수 (간략화 버전)
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

def apply_nms(boxes, scores, iou_threshold=0.5):
    # 1. 점수(Confidence)가 높은 순서대로 박스 번호(index) 정렬
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep_boxes = []
    
    while sorted_indices:
        # 2. 현재 가장 점수 높은 박스 확정 (리스트에서 빼내서 저장)
        current_idx = sorted_indices.pop(0)
        keep_boxes.append(current_idx)
        
        # 3. 확정된 박스와 남은 박스들의 IoU를 비교
        # 기준치(threshold) 이상 겹치면 중복으로 간주하고 버림
        indices_to_keep = []
        for idx in sorted_indices:
            iou = calculate_iou(boxes[current_idx], boxes[idx])
            if iou < iou_threshold: 
                indices_to_keep.append(idx) # 안 겹치는 애들만 살림
                
        sorted_indices = indices_to_keep
        
    return keep_boxes

def test_nms():
    # 가상의 예측 박스 [x1, y1, x2, y2] 와 그 박스의 신뢰도 점수
    boxes = [
        [50, 50, 150, 150],   # 박스 A (점수 0.9, 확정)
        [55, 55, 145, 145],   # 박스 B (점수 0.75, 박스 A와 엄청 겹침 -> 지워져야 함)
        [200, 200, 300, 300]  # 박스 C (점수 0.8, 아예 다른 곳에 있음 -> 살아남아야 함)
    ]
    scores = [0.9, 0.75, 0.8] 
    
    print("=== NMS (Non-Maximum Suppression) Test ===")
    print(f"Before NMS - Total boxes: {len(boxes)}")
    
    # NMS 실행 (IoU가 0.5 이상 겹치면 제거)
    kept_indices = apply_nms(boxes, scores, iou_threshold=0.5)
    
    print("\nAfter NMS - Final kept boxes:")
    for idx in kept_indices:
        print(f" - Box {idx}: {boxes[idx]} (Score: {scores[idx]})")

if __name__ == "__main__":
    test_nms()