import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 커스텀 데이터셋 (가상의 64x64 이미지 100장)
class DummyVisionDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(100, 3, 64, 64) 
        self.y = torch.randint(0, 2, (100,)) # 0 또는 1 (이진 분류)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 2. 합성곱 신경망(CNN) 모델
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # 64x64 이미지가 16채널을 통과한 후 1차원으로 펴짐
        self.fc = nn.Linear(16 * 64 * 64, 2) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_pipeline():
    # 3. 데이터로더, 모델, 손실함수, 옵티마이저 세팅
    dataset = DummyVisionDataset()
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss() # 분류 문제용 손실함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("=== Full Pipeline Training Start ===")
    
    # 4. 전체 학습 루프
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # 예측 및 오차 계산
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 모델 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 에포크마다 평균 오차 출력
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
    print("=== Training Complete! ===")

if __name__ == "__main__":
    train_pipeline()