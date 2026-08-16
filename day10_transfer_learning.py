import torch
import torch.nn as nn
import torchvision.models as models

def custom_transfer_learning():
    # 사전학습 ResNet18 로드
    model = models.resnet18(weights='DEFAULT')
    
    # 기존 FC 레이어 입력 특징 수 확인
    num_ftrs = model.fc.in_features
    
    # 전이학습: 최종 출력층을 2개 클래스용으로 덮어쓰기
    model.fc = nn.Linear(num_ftrs, 2)
    
    # 더미 입력 텐서 (1장, RGB, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 추론 모드 (기울기 계산 X)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print("=== Transfer Learning Model Test ===")
    print(f"Original FC in_features: {num_ftrs}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Modified Output shape: {output.shape}") # 1000 -> 2 변경 확인

if __name__ == "__main__":
    custom_transfer_learning()