from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt

# 1. Tạo dataset giả lập
X, y = make_blobs(
    n_samples=500,      # số điểm dữ liệu
    centers=3,          # số cụm thật
    n_features=2,       # số đặc trưng, ở đây là 2 để vẽ được trên mặt phẳng
    cluster_std=0.8,    # độ phân tán của các cụm
    random_state=42     # giúp kết quả cố định mỗi lần chạy
)

# 2. Đưa dữ liệu vào DataFrame
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])

# Thêm nhãn thật để tham khảo
df["True Label"] = y

# 3. Xem dữ liệu
print("5 dòng đầu của dataset:")
print(df.head())

print("\nKích thước dataset:")
print(df.shape)

# 4. Lưu ra file CSV
df.to_csv("cluster_data_make_blobs.csv", index=False)

print("\nĐã lưu file: cluster_data_make_blobs.csv")

# 5. Vẽ dữ liệu
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Feature 1"],
    df["Feature 2"],
    c=df["True Label"],
    s=40
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset tạo bằng make_blobs")
plt.grid(True)
plt.show()