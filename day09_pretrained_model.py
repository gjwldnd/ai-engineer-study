import torch
import torchvision.models as models

def run_pretrained_resnet():
    model = models.resnet18(weights='DEFAULT')
    
    model.eval() 

    #  가짜 이미지 텐서 생성 
    dummy_input = torch.randn(1, 3, 224, 224)

    #  모델 추론 진행 
    with torch.no_grad():
        output = model(dummy_input)

    print("=== Pre-trained Model Inference Start ===")
    print(f"Input shape: {dummy_input.shape}")
    
    print(f"Output shape: {output.shape}")
    print("=== Inference Complete! ===")

if __name__ == "__main__":
    run_pretrained_resnet()