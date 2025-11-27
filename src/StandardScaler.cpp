#include "../include/StandardScaler.h"

// Helper: Chuyển đổi an toàn variant sang double
double StandardScaler::get_value_as_double(const DataType& val) {
    return visit([](auto&& arg) -> double {
        using T = decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, string>) return 0.0; // Nếu là string thì trả về 0 (hoặc xử lý lỗi)
        else return static_cast<double>(arg);
    }, val);
}

void StandardScaler::fit(const Dataframe& X) {
    mean_map.clear();
    std_map.clear();
    size_t n = X.nRows;
    
    // Duyệt qua từng cột
    for (const auto& col : X.data) {
        const string& col_name = col.first;
        const vector<DataType>& feature = col.second;

        double mean = 0;
        double std_dev = 0;

        // 1. Tính Mean
        for (int i = 0; i < n; i++) {
            mean += get_value_as_double(feature.at(i));
        }
        mean = mean / n;

        // 2. Tính Std (Standard Deviation)
        for (int i = 0; i < n; i++) {
            double val = get_value_as_double(feature.at(i));
            std_dev += (val - mean) * (val - mean);
        }
        
        // Tránh chia cho 0 nếu n=0
        if (n > 0) {
            std_dev = sqrt(std_dev / n);
        }

        // Nếu std = 0 (cột hằng số), gán bằng 1 để tránh chia cho 0 khi transform
        if (std_dev == 0) std_dev = 1.0;

        mean_map[col_name] = mean;
        std_map[col_name] = std_dev;
    }
}

void StandardScaler::transform(Dataframe& X) {
    size_t n = X.nRows;
    
    // Duyệt qua các cột trong Dataframe
    for (auto& col : X.data) {
        string col_name = col.first;
        
        // Chỉ chuẩn hóa nếu cột đó đã được fit (có trong mean_map)
        if (mean_map.find(col_name) != mean_map.end()) {
            double mean = mean_map[col_name];
            double std_dev = std_map[col_name];
            auto& feature = col.second;

            for (int i = 0; i < n; i++) {
                double old_val = get_value_as_double(feature.at(i));
                // Công thức Z-score: z = (x - u) / s
                feature.at(i) = (old_val - mean) / std_dev;
            }
        }
    }
}

void StandardScaler::fit_transform(Dataframe& X) {
    fit(X);      // Tính toán thống kê
    transform(X); // Áp dụng chuẩn hóa
}