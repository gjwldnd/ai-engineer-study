import torch
import numpy as np

def image_tensor_practice():
    print("=== 1. 가상의 컬러 이미지 텐서 생성 (PyTorch) ===")
    # 크기가 256x256이고, RGB 3채널을 가진 이미지라고 가정 (C, H, W)
    # 0.0 ~ 1.0 사이의 무작위 픽셀 값을 가진 텐서 생성
    pytorch_image = torch.rand(3, 256, 256)
    print("PyTorch 이미지 형태 (C, H, W):", pytorch_image.shape)

    print("\n=== 2. PyTorch 텐서를 NumPy 배열로 변환 ===")
    # 텐서를 넘파이 배열로 변환
    numpy_image = pytorch_image.numpy()
    print("NumPy 변환 직후 형태:", numpy_image.shape)

    print("\n=== 3. 차원 순서 변경 (C, H, W) -> (H, W, C) ===")
    # 이미지 시각화 라이브러리들은 주로 (H, W, C)를 요구하므로 순서를 바꿔줍니다.
    # np.transpose()를 사용해 0, 1, 2번째 차원의 순서를 1, 2, 0으로 변경
    transposed_image = np.transpose(numpy_image, (1, 2, 0))
    print("시각화용 이미지 형태 (H, W, C):", transposed_image.shape)

if __name__ == "__main__":
    image_tensor_practice()