Cài đặt thuật toán Naive Bayes Phân loại hành vi truy cập mạng
Dự án này tập trung vào việc tự cài đặt thuật toán Gaussian Naive Bayes (GaussianNB) bằng ngôn ngữ C++ nhằm phân tích và định lượng rủi ro bảo mật trong các luồng giao tiếp mạng. Mục tiêu cốt lõi là xây dựng một hệ thống phân loại hoàn chỉnh từ các nguyên lý toán học cơ bản mà không phụ thuộc vào các thư viện học máy có sẵn.
1. Giới thiệu tổng quan
Trong bối cảnh các cuộc tấn công mạng ngày càng phức tạp, dự án sử dụng lý thuyết xác suất Bayes để ước lượng khả năng xảy ra tấn công và mức độ rủi ro cho từng thành phần trong hệ thống một cách tự động. Sản phẩm đầu ra là một chương trình C++ có khả năng huấn luyện mô hình dựa trên các tập dữ liệu đã được làm sạch.
2. Các lớp thiết kế chính
• Dataframe: Đọc và quản lý dữ liệu từ tệp CSV, tổ chức dưới dạng vector hai chiều để tối ưu truy xuất.
• StandardScaler: Thực hiện chuẩn hóa Z-score (đưa dữ liệu về trung bình bằng 0 và độ lệch chuẩn bằng 1).
• TrainTestSplit: Phân tách dữ liệu thành tập huấn luyện và tập kiểm thử theo tỉ lệ, hỗ trợ cơ chế xáo trộn ngẫu nhiên và phân lớp tầng (Stratified Split).
• GaussianNB: Triển khai thuật toán Gaussian Naive Bayes, sử dụng kỹ thuật log-likelihood để tăng độ ổn định số học và tránh hiện tượng tràn số (underflow).
• Metrics: Đánh giá hiệu năng mô hình thông qua các chỉ số: Accuracy, Precision, Recall, F1-Score và Ma trận nhầm lẫn (Confusion Matrix).
3. Quy trình thực hiện (Pipeline)
Luồng hoạt động của chương trình bao gồm các bước tuần tự:
    1. Đầu vào: Tải dữ liệu từ tệp CSV.
    2. Xử lý: Chuẩn hóa dữ liệu qua StandardScaler và phân tách tập Train-Test.
    3. Huấn luyện: Mô hình tính toán giá trị trung bình và phương sai cho từng đặc trưng theo từng nhãn lớp.
    4. Dự đoán: Áp dụng công thức Bayes kết hợp hàm mật độ xác suất Gaussian để đưa ra nhãn dự đoán.
    5. Đánh giá: So sánh kết quả dự đoán với nhãn thực tế.
4. Dữ liệu sử dụng
Dự án đã thực nghiệm trên hai bộ dữ liệu lớn:
• CICIDS2017: Từ 78 đặc trưng ban đầu, dữ liệu được rút gọn xuống 48 đặc trưng quan trọng nhất và gộp thành 8 nhãn chính (như DoS, PortScan, Brute Force, Web Attack, Bot...).
• CIC IoT 2023: Bộ dữ liệu với hơn 20 triệu bản ghi trong môi trường Internet of Things để kiểm tra khả năng tổng quát hóa của mô hình.
5. Hiệu năng thực nghiệm
• Tốc độ huấn luyện: Phiên bản C++ (0.44s) nhanh hơn đáng kể so với thư viện Scikit-learn của Python (1.87s) trên cùng một tập dữ liệu.
• Độ chính xác: Đạt mức tương đương với mô hình chuẩn của Scikit-learn (khoảng 50% trên CICIDS2017 và 75% trên CIC IoT 2023).
6. Hướng dẫn sử dụng
Yêu cầu hệ thống
• Trình biên dịch C++ hỗ trợ chuẩn C++17 trở lên (g++ hoặc clang++).
• Hệ điều hành: Windows/Linux.
Cách chạy chương trình
1. Clone dự án hoặc tải tệp model.rar từ GitHub.
2. Giải nén và khởi chạy file thực thi model.exe.
3. Thao tác trên giao diện terminal:
    ◦ Nhập đường dẫn file CSV (ví dụ: E:/Dataset/datafinal.csv).
    ◦ Nhập tên cột nhãn (Label).
    ◦ Nhập tỷ lệ tập Test (ví dụ: 30 cho 30%).
