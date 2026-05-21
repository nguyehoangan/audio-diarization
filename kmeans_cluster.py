import pandas as pd #thư viện để xử lý file CSV 
import matplotlib.pyplot as plt #thư viện để vẽ biểu đồ

from sklearn.cluster import KMeans #thư viện để thực hiện thuật toán K-Means
from sklearn.preprocessing import StandardScaler #thư viện để chuẩn hóa dữ liệu (đưa về cùng thang đo)


# 1. Đọc dữ liệu
file_path = r"C:\Users\Admin\Downloads\cluster_data.csv" #chuyển file này vào cùng thư mục với file code này thì chỉ cần "cluster_data.csv"

df = pd.read_csv(file_path) #đọc dữ liệu từ file CSV vào DataFrame

print("5 dòng đầu của dữ liệu:")
print(df.head())

print("\nKích thước dữ liệu:", df.shape)
print("\nThông tin dữ liệu:")
print(df.info())


# 2. Lấy các cột đặc trưng
X = df[["Feature 1", "Feature 2"]] 


# 3. Chuẩn hóa dữ liệu (do K-Means dùng khoảng cách để phân cụm, nên nếu 1 cột có giá trị rất lớn so với cột còn lại thì sẽ ảnh hưởng đến kết quả phân cụm, nên cần chuẩn hóa dữ liệu để đưa về cùng thang đo)
scaler = StandardScaler() #tạo bộ chuẩn hóa dữ liệu
X_scaled = scaler.fit_transform(X) 
#fit: tính toán trung bình và độ lệch chuẩn; transform: chuẩn hóa dữ liệu dựa trên công thức (x_scaled = (x - mean) / standard_deviation). Sau khi chuẩn hóa, dữ liệu sẽ có trung bình bằng 0 và độ lệch chuẩn bằng 1 (phân phối chuẩn)
#tạo ra 1 mảng numpy chứa dữ liệu đã được chuẩn hóa, có cùng số hàng và số cột như X nhưng giá trị đã được điều chỉnh để có cùng thang đo.

# 4. Dùng Elbow Method để chọn K
wcss = []

K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(        #tạo mô hình K-Means với số cụm k, random_state để đảm bảo kết quả cố định mỗi lần chạy, n_init là số lần thuật toán sẽ chạy với các khởi tạo khác nhau để chọn ra kết quả tốt nhất
        n_clusters=k,
        random_state=42,    #chọn centroid ngẫu nhiên
        n_init=10
    )
    kmeans.fit(X_scaled)    #chạy thuật toán K-Means trên dữ liệu đã chuẩn hóa
    wcss.append(kmeans.inertia_) #lưu giá trị WCSS vào wcss

#vẽ biểu đồ Elbow để chọn số cụm K
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker="o")
plt.xlabel("Số cụm K")
plt.ylabel("WCSS")
plt.title("Elbow Method để chọn số cụm K")
plt.grid(True)
plt.show()


# 5. Chạy K-Means
# Sau khi xem biểu đồ Elbow, bạn có thể đổi giá trị K
K = 3

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled) #trả về nhãn cụm cho từng điểm dữ liệu

df["Cluster"] = labels #thêm cột "Cluster" vào DataFrame để lưu nhãn cụm


# 6. Lấy tọa độ centroid
centers_scaled = kmeans.cluster_centers_ #lấy tọa độ các centroid sau khi K-Means chạy xong (trong không gian đã chuẩn hóa)

# Chuyển centroid từ dữ liệu đã chuẩn hóa về dữ liệu gốc
centers = scaler.inverse_transform(centers_scaled)

print("\nTâm cụm centroid:")
for i, center in enumerate(centers):
    print(f"Cụm {i}: Feature 1 = {center[0]:.3f}, Feature 2 = {center[1]:.3f}")


# 7. Vẽ kết quả phân cụm
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Feature 1"],
    df["Feature 2"],
    c=df["Cluster"],
    s=40
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    label="Centroid"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"K-Means Clustering với K = {K}")
plt.legend()
plt.grid(True)
plt.show()


# 8. Lưu kết quả ra file mới
df.to_csv("cluster_data_result.csv", index=False) #nếu không có index=False thì sẽ thêm cột thứ tự không cần thiết vào file CSV

print("\nĐã lưu kết quả vào file: cluster_data_result.csv")
print(df.head())