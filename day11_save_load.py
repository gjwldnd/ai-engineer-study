import torch
import torch.nn as nn
import torchvision.models as models
import os

def save_and_load_model():
    # 1. 원본 모델 세팅 
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # 2. 모델 가중치 저장 
    save_path = "my_custom_model.pth"
    torch.save(model.state_dict(), save_path)
    
    print("=== Model Save ===")
    print(f"Saved path: {os.path.abspath(save_path)}")

    # 3. 새로운 빈 모델 생성 및 구조 맞추기
    loaded_model = models.resnet18() 
    loaded_model.fc = nn.Linear(loaded_model.fc.in_features, 2) 
    
    # 4. 저장된 가중치 불러와서 덮어쓰기
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval() # 추론 모드 전환
    
    print("\n=== Model Load & Test ===")
    
    # 더미 데이터 추론 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = loaded_model(dummy_input)
        
    print(f"Loaded Model Output shape: {output.shape}")

if __name__ == "__main__":
    save_and_load_model()