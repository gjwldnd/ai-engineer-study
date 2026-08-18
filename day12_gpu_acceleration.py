import torch
import torch.nn as nn
import torchvision.models as models

def check_and_use_gpu():
    # 1. 사용 가능한 디바이스자동 판별
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"=== Selected Device: {device} ===")

    # 2. 모델 생성 및 디바이스(GPU)로 이동
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device) 

    # 3. 더미 데이터 생성 및 디바이스(GPU)로 이동
    # 모델과 데이터가 같은 디바이스에 있어야 에러가 안 남
    dummy_input = torch.randn(16, 3, 224, 224).to(device)

    # 4. 추론 테스트
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print("\n=== GPU Inference Test ===")
    print(f"Data device: {dummy_input.device}")
    print(f"Output shape: {output.shape}") 

if __name__ == "__main__":
    check_and_use_gpu()