import torch
import numpy as np

def tensor_exploration():
    print("--- 1. 텐서 생성하기 ---")
    empty_tensor = torch.zeros(3, 3)
    rand_tensor = torch.rand(2, 2)
    print("Zero Tensor:\n", empty_tensor)
    print("Random Tensor:\n", rand_tensor)

    print("\n--- 2. 리스트와 Numpy 배열을 텐서로 변환 ---")
    my_list = [[1, 2], [3, 4]]
    tensor_from_list = torch.tensor(my_list)
    print("Tensor from list:\n", tensor_from_list)

    print("\n--- 3. 텐서의 기본 연산 ---")
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    print("a + b =", a + b)
    print("a * b =", a * b)

if __name__ == "__main__":
    tensor_exploration()