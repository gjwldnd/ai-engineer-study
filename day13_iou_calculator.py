def calculate_iou(box1, box2):
    # 박스 형태: [x1, y1, x2, y2] (좌상단, 우하단 좌표)
    
    # 1. 겹치는 영역(Intersection)의 좌표 계산
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 2. 겹치는 영역 넓이 계산 (안 겹치면 0)
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    # 3. 각 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 4. 합집합(Union) 넓이 및 IoU 계산
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def test_iou():
    # 정답 박스(Ground Truth)와 예측 박스(Prediction) 가상 데이터
    gt_box = [50, 50, 150, 150]
    pred_box = [70, 80, 170, 160]
    
    iou_score = calculate_iou(gt_box, pred_box)
    
    print("=== Object Detection IoU Test ===")
    print(f"Ground Truth Box: {gt_box}")
    print(f"Prediction Box:   {pred_box}")
    print(f"IoU Score: {iou_score:.4f}")

if __name__ == "__main__":
    test_iou()