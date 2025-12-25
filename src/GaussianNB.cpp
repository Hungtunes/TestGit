/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Thực thi lớp NaiveBayes dưới biến thể GausianNB
*/

#include "../include/NaiveBayes.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>

const double PI = 3.14159265358979323846;

// Hàm inline giúp tính nhanh xác suất dưới dạng log (tránh lỗi tràn số dưới)
inline double calculate_log_pdf(double x, double mean, double var) {
    double diff = x - mean;
    return -0.5 * (log(2.0 * PI * var) + (diff * diff) / var);
}


// Phương thức huấn luyện: Học các đặc trưng thống kê từ tập dữ liệu Train
void GaussianNB::fit(const Dataframe& X, const Dataframe& y) {
    classes.clear();
    priors.clear();
    mean_map.clear();
    var_map.clear();

    string label_col = y.cols[0];
    const auto& y_data = y.data.at(label_col);
    size_t n_rows = X.nRows;

    // Phân loại các hàng dữ liệu theo từng nhãn (Label)
    map<string, vector<int>> class_indices;
    for (size_t i = 0; i < n_rows; ++i) {
        string label = visit([](auto&& arg) -> string {
             using T = decay_t<decltype(arg)>;
             if constexpr (is_same_v<T, string>) return arg;
             else return to_string(arg);
        }, y_data[i]);
        class_indices[label].push_back(i);
    }

    double global_max_var = 0.0; 

    // Duyệt qua từng lớp để tính toán các thông số xác suất
    for (auto const& [cls, indices] : class_indices) {
        classes.push_back(cls);
        double n_samples_in_class = indices.size();

        // ính xác suất tiên nghiệm (Prior probability)
        priors[cls] = n_samples_in_class / static_cast<double>(n_rows);

        for (const string& col_name : X.cols) {
            double sum = 0;
            const auto& col_data = X.data.at(col_name);
            
            // Tính giá trị trung bình (Mean) của đặc trưng theo từng lớp
            for (int idx : indices) {
                visit([&sum](auto&& arg) {
                    using T = decay_t<decltype(arg)>;
                    if constexpr (!is_same_v<T, string>) sum += static_cast<double>(arg);
                }, col_data[idx]);
            }
            double mean = sum / n_samples_in_class;
            mean_map[cls][col_name] = mean;

            // Tính phương sai (Variance)
            double var_sum = 0;
            for (int idx : indices) {
                visit([&var_sum, mean](auto&& arg) {
                    using T = decay_t<decltype(arg)>;
                    if constexpr (!is_same_v<T, string>) {
                        double val = static_cast<double>(arg);
                        var_sum += (val - mean) * (val - mean);
                    }
                }, col_data[idx]);
            }
            double var = var_sum / n_samples_in_class;
            
            // Tìm phương sai lớn nhất để tính hệ số làm mịn (epsilon)
            if (var > global_max_var) {
                global_max_var = var;
            }
            
            var_map[cls][col_name] = var;
        }
    }

    // Thêm một lượng nhỏ epsilon để tránh lỗi chia cho 0 khi phương sai bằng 0
    double epsilon = 1e-9 * global_max_var; 

    // Cập nhật lại toàn bộ phương sai với hệ số làm mịn
    for (auto const& cls : classes) {
        for (const string& col_name : X.cols) {
            var_map[cls][col_name] += epsilon;
        }
    }
}

// Phương thức dự đoán: Tính toán và chọn lớp có xác suất cao nhất cho dữ liệu mới
vector<string> GaussianNB::predict(const Dataframe& X) {
    size_t n_rows = X.nRows;
    size_t n_classes = classes.size();
    
    // Khởi tạo bảng điểm với giá trị Log Prior (điểm bắt đầu của mỗi lớp)
    vector<vector<double>> scores(n_rows, vector<double>(n_classes, 0.0));

    for (size_t c = 0; c < n_classes; ++c) {
        double log_prior = log(priors[classes[c]]);
        for (size_t i = 0; i < n_rows; ++i) {
            scores[i][c] = log_prior;
        }
    }

    // Tính toán Log Likelihood cho từng đặc trưng của mẫu dữ liệu
    for (size_t c = 0; c < n_classes; ++c) {
        string cls = classes[c];
        
        for (const string& col_name : X.cols) {
            double mean = mean_map[cls][col_name];
            double var = var_map[cls][col_name];
  
            // Tính toán trước các hằng số để tối ưu tốc độ vòng lặp
            double log_var_term = -0.5 * log(2.0 * PI * var); 
            double inv_two_var = 1.0 / (2.0 * var);

            const auto& col_data = X.data.at(col_name); 

            for (size_t i = 0; i < n_rows; ++i) {
                double x = 0.0;
                visit([&x](auto&& arg) {
                    using T = decay_t<decltype(arg)>;
                    if constexpr (!is_same_v<T, string>) x = static_cast<double>(arg);
                }, col_data[i]);

                // Áp dụng công thức Gaussian PDF ở dạng Log
                double diff = x - mean;
                scores[i][c] += log_var_term - (diff * diff) * inv_two_var;
            }
        }
    }

    // Trích xuất kết quả cuối cùng: Chọn nhãn có điểm số cao nhất (Argmax)
    vector<string> predictions;
    predictions.reserve(n_rows);

    for (size_t i = 0; i < n_rows; ++i) {
        double max_score = -numeric_limits<double>::infinity();
        int best_idx = 0;
        for (size_t c = 0; c < n_classes; ++c) {
            if (scores[i][c] > max_score) {
                max_score = scores[i][c];
                best_idx = c;
            }
        }
        predictions.push_back(classes[best_idx]);
    }

    return predictions;
}