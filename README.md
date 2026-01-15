# Cài đặt thuật toán Gaussian Naive Bayes bằng C++

Dự án này tập trung vào việc **tự cài đặt thuật toán Gaussian Naive Bayes (GaussianNB)** bằng ngôn ngữ **C++**, nhằm **phân loại hành vi truy cập mạng** và **định lượng rủi ro bảo mật** trong các luồng giao tiếp mạng. Toàn bộ mô hình được xây dựng **từ các nguyên lý toán học cơ bản**, **không phụ thuộc vào các thư viện học máy có sẵn**.

---

## 1. Giới thiệu tổng quan

Trong bối cảnh các cuộc tấn công mạng ngày càng tinh vi, dự án áp dụng **lý thuyết xác suất Bayes** để ước lượng khả năng xảy ra tấn công và mức độ rủi ro của từng luồng mạng một cách tự động.

Sản phẩm đầu ra là một **chương trình C++ hoàn chỉnh**, có khả năng huấn luyện và đánh giá mô hình dựa trên các tập dữ liệu mạng đã được làm sạch.

---

## 2. Các lớp thiết kế chính

### 2.1. `Dataframe`

* Đọc và quản lý dữ liệu từ tệp CSV.
* Lưu trữ dữ liệu dưới dạng `vector<vector<double>>` để tối ưu truy xuất và tính toán.

### 2.2. `StandardScaler`

* Thực hiện chuẩn hóa **Z-score**:

  * Đưa mỗi đặc trưng về trung bình bằng 0.
  * Độ lệch chuẩn bằng 1.

### 2.3. `TrainTestSplit`

* Phân tách dữ liệu thành tập **Train / Test** theo tỷ lệ chỉ định.
* Hỗ trợ:

  * Xáo trộn ngẫu nhiên (Shuffle).
  * Phân lớp tầng (**Stratified Split**) để giữ nguyên phân bố nhãn.

### 2.4. `GaussianNB`

* Cài đặt thuật toán **Gaussian Naive Bayes** từ đầu.
* Tính toán:

  * Xác suất tiên nghiệm của từng lớp.
  * Trung bình và phương sai cho từng đặc trưng theo từng lớp.
* Sử dụng **log-likelihood** nhằm:

  * Tăng độ ổn định số học.
  * Tránh hiện tượng tràn số (underflow).

### 2.5. `Metrics`

* Đánh giá hiệu năng mô hình thông qua các chỉ số:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * Confusion Matrix

---

## 3. Quy trình thực hiện (Pipeline)

Luồng xử lý của chương trình bao gồm các bước tuần tự:

1. **Đầu vào**: Tải dữ liệu từ tệp CSV.
2. **Tiền xử lý**:

   * Chuẩn hóa dữ liệu bằng `StandardScaler`.
   * Chia tập Train / Test bằng `TrainTestSplit`.
3. **Huấn luyện**:

   * Tính toán trung bình và phương sai cho từng đặc trưng theo từng nhãn lớp.
4. **Dự đoán**:

   * Áp dụng công thức Bayes.
   * Sử dụng hàm mật độ xác suất Gaussian để suy ra nhãn dự đoán.
5. **Đánh giá**:

   * So sánh nhãn dự đoán với nhãn thực tế.
   * Xuất các chỉ số đánh giá và ma trận nhầm lẫn.

---

## 4. Dữ liệu sử dụng

### 4.1. CICIDS2017

* Ban đầu gồm **78 đặc trưng**.
* Được rút gọn xuống **48 đặc trưng quan trọng nhất**.
* Gộp thành **8 nhãn chính**, bao gồm:

  * Benign
  * DoS / DDoS
  * PortScan
  * Brute Force
  * Web Attack
  * Bot
  * Reconnaissance
  * Spoofing

### 4.2. CIC IoT 2023

* Bộ dữ liệu quy mô lớn với **hơn 20 triệu bản ghi**.
* Mô phỏng môi trường **Internet of Things (IoT)**.
* Dùng để kiểm tra khả năng **tổng quát hóa (generalization)** của mô hình.

---

## 5. Hiệu năng thực nghiệm

### 5.1. Tốc độ huấn luyện

| Mô hình                   | Thời gian huấn luyện |
| ------------------------- | -------------------- |
| GaussianNB (C++)          | 0.44 giây            |
| GaussianNB (Scikit-learn) | 1.87 giây            |

Phiên bản C++ cho tốc độ nhanh hơn đáng kể trên cùng một tập dữ liệu.

### 5.2. Độ chính xác

* CICIDS2017: khoảng **50%**
* CIC IoT 2023: khoảng **75%**

Hiệu năng đạt mức tương đương với mô hình chuẩn của **Scikit-learn**.

---

## 6. Hướng dẫn sử dụng

### 6.1. Yêu cầu hệ thống

* Trình biên dịch hỗ trợ **C++17** trở lên (`g++`, `clang++`).
* Hệ điều hành: **Windows**

### 6.2. Cách chạy chương trình

1. Clone dự án hoặc tải file `model.rar` từ GitHub.
2. Giải nén và chạy file thực thi `model.exe`.
3. Thao tác trên giao diện terminal:

   * Nhập đường dẫn tới file CSV (ví dụ: `E:/Dataset/datafinal.csv`).
   * Nhập tên cột nhãn (ví dụ: `Label`).
   * Nhập tỷ lệ tập Test (ví dụ: `30` cho 30%).

---
