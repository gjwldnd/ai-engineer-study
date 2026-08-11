import torch
import torch.nn as nn

class SimpleVisionModel(nn.Module):
    def __init__(self):
        super(SimpleVisionModel, self).__init__()
        # Conv Layer: RGB(3) 이미지를 받아 16개의 특징 맵으로 추출
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward Pass
        x = self.conv_layer(x)
        x = self.relu(x)
        return x

def test_model():
    model = SimpleVisionModel()
    print("=== Model Structure ===")
    print(model)

    # 가상의 입력 데이터 생성: (Batch_size, Channels, Height, Width)
    dummy_image = torch.rand(1, 3, 224, 224) 
    print(f"\nInput shape: {dummy_image.shape}")
    
    # 모델 추론 (통과)
    output = model(dummy_image)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    test_model()