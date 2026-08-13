import torch
from torch.utils.data import Dataset, DataLoader

class CustomImageDataset(Dataset):
    def __init__(self):
        # 1. 가상의 이미지 데이터와 정답(라벨) 준비 (총 100장)
        self.x_data = torch.randn(100, 3, 224, 224) 
        self.y_data = torch.randint(0, 2, (100,)) # 0 또는 1 (고양이 or 강아지 등)

    def __len__(self):
        # 2. 전체 데이터의 총 개수 반환
        return len(self.x_data)

    def __getitem__(self, idx):
        # 3. 모델이 데이터를 요청할 때, 특정 인덱스(idx)의 데이터 1개를 반환
        return self.x_data[idx], self.y_data[idx]

def test_dataloader():
    # 4. 데이터셋 객체 생성
    dataset = CustomImageDataset()

    # 5. 데이터로더 세팅: 100장의 데이터를 한 번에 16장(batch_size)씩 무작위로 섞어서(shuffle) 공급
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    print("=== DataLoader Test Start ===")
    
    # 6. 데이터로더 동작 확인 (학습 루프에서 데이터를 받아오는 과정)
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f" - Images shape: {images.shape}")
        print(f" - Labels shape: {labels.shape}")
        
        # 전체 흐름만 확인하기 위해 첫 번째 배치(16장)만 출력하고 반복문 종료
        if batch_idx == 0:
            break

if __name__ == "__main__":
    test_dataloader()