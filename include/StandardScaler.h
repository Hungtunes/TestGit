#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include "Dataframe.h" // Import Dataframe để sử dụng

using namespace std;

class StandardScaler {
public:
    // Lưu trữ Mean và Std của từng cột
    unordered_map<string, double> mean_map;
    unordered_map<string, double> std_map;

    // Tính toán Mean và Std từ dữ liệu (thường là tập Train)
    // Dùng const Dataframe& vì fit không làm thay đổi dữ liệu gốc
    void fit(const Dataframe& X);

    // Áp dụng công thức chuẩn hóa: (x - u) / s
    // Hàm này làm thay đổi trực tiếp Dataframe truyền vào
    void transform(Dataframe& X);

    // Kết hợp cả 2 bước: Tính toán xong chuẩn hóa luôn
    void fit_transform(Dataframe& X);

private:
    // Helper: Lấy giá trị double an toàn từ variant (xử lý cả int và double)
    double get_value_as_double(const DataType& val);
};