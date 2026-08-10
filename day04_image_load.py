import cv2
import matplotlib.pyplot as plt

def process_image(image_path):
    print("=== 1. OpenCV로 이미지 불러오기 ===")
    img = cv2.imread(image_path)
    
    if img is None:
        print("에러: 이미지를 찾을 수 없습니다. 파일 이름과 위치를 확인해주세요!")
        return

    print("원본 이미지 크기 (H, W, C):", img.shape)

    print("\n=== 2. 색상 공간 변환 (BGR -> RGB) ===")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("\n=== 3. 딥러닝 모델용으로 크기 조절 (Resizing) ===")
    resized_img = cv2.resize(img_rgb, (224, 224))
    print("조절된 이미지 크기:", resized_img.shape)

    print("\n=== 4. 화면에 이미지 띄우기 ===")
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    
    plt.subplot(1, 2, 2)
    plt.title("Resized Image (224x224)")
    plt.imshow(resized_img)
    
    plt.show()

if __name__ == "__main__":
    process_image("C:/python/ai-engineer-study/sample.jpg")