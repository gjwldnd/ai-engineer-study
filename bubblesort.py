def bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

if __name__ == "__main__":
    sample_list = [64, 34, 25, 12, 22, 11, 90]
    print("정렬 전:", sample_list)
    
    sorted_list = bubble_sort(sample_list)
    print("정렬 후:", sorted_list)