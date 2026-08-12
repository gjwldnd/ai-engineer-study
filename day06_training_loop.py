import torch
import torch.nn as nn
import torch.optim as optim

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1) # 단순 선형 모델 (입력 1, 출력 1)

    def forward(self, x):
        return self.linear(x)

def train_model():
    # 1. 학습 데이터 준비 (y = 2x 패턴)
    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    # 2. 모델, 손실함수(오차 계산), 옵티마이저(학습 도구) 세팅
    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print("=== Training Start ===")
    
    # 3. 실제 학습 루프 (100번 반복)
    epochs = 100
    for epoch in range(epochs):
        # 예측 및 현재 오차 확인
        prediction = model(x_train)
        loss = criterion(prediction, y_train)

        # 오차를 바탕으로 모델 가중치 업데이트 (학습 진행)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. 학습 완료 후 잘 맞추는지 테스트
    print("\n=== Test after Training ===")
    test_data = torch.tensor([[5.0]])
    predicted_result = model(test_data)
    print(f"Input: 5.0 -> Predicted: {predicted_result.item():.4f} (Expected: 10.0)")

if __name__ == "__main__":
    train_model()