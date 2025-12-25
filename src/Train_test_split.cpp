/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Thực thi lớp TrainTestSplit
*/

#include "../include/Train_test_split.h"
#include <algorithm>
#include <random>    
#include <iostream>
#include <variant>  

// Chuyển đổi an toàn variant sang String để xử lý nhãn (Label)
string Train_test_split::variant_to_string(const DataType& val) {
    return visit([](auto&& arg) -> string {
        using T = decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, string>) return arg;
        else return to_string(arg);
    }, val);
}

// Trích xuất một tập dữ liệu con (Subset) dựa trên danh sách chỉ số 
Dataframe Train_test_split::create_subset(const Dataframe& original, const vector<int>& indices) {
    Dataframe sub;
    sub.cols = original.cols;
    sub.nCols = original.nCols;
    sub.nRows = indices.size();

    // 1. Khởi tạo và cấp phát trước bộ nhớ (Tối ưu performance)
    for (const string& col_name : sub.cols) {
        sub.data[col_name].reserve(indices.size());
    }

    // 2. Tiến hành copy dữ liệu từ bản gốc sang tập con
    for (const string& col_name : sub.cols) {
        const auto& src_vec = original.data.at(col_name);
        auto& dest_vec = sub.data[col_name];
        for (int idx : indices) {
            dest_vec.push_back(src_vec[idx]);
        }
    }
    return sub;
}

// Phương thức chính: Chia dữ liệu thành tập Train và Test
vector<Dataframe> Train_test_split::train_test_split(const Dataframe& X, const Dataframe& y, int test_size, bool stratify) {
    
    // BƯỚC 1: Kiểm tra tính khớp của dữ liệu đầu vào
    if (X.nRows != y.nRows) {
        cout << "❌ Lỗi: Số lượng hàng của X và y không khớp!" << endl;
        return {};
    }

    size_t total_rows = X.nRows;
    vector<int> train_indices;
    vector<int> test_indices;
    
    // Sử dụng seed cố định (42) để kết quả chia giống nhau ở mỗi lần chạy (Reproducibility)
    mt19937 g(42); 

    // BƯỚC 2: Lựa chọn kỹ thuật chia dữ liệu
    if (stratify && y.nCols > 0) {
        // --- CÁCH A: CHIA THEO TỶ LỆ LỚP (STRATIFIED) ---
        map<string, vector<int>> class_groups;
        string label_col = y.cols[0];
        const auto& label_data = y.data.at(label_col);

        // Gom nhóm chỉ số theo từng lớp nhãn
        for (size_t i = 0; i < total_rows; ++i) {
            string key = variant_to_string(label_data[i]);
            class_groups[key].push_back(i);
        }

        // Chia nhỏ từng nhóm lớp theo tỷ lệ test_size
        for (auto& [key, indices] : class_groups) {
            shuffle(indices.begin(), indices.end(), g);
            
            size_t n_test_local = (indices.size() * test_size) / 100;
            // Đảm bảo mỗi lớp có ít nhất 1 mẫu trong tập test nếu có đủ dữ liệu
            if (n_test_local == 0 && indices.size() > 1 && test_size > 0) n_test_local = 1;

            size_t split_point = indices.size() - n_test_local;

            for (size_t i = 0; i < split_point; ++i) train_indices.push_back(indices[i]);
            for (size_t i = split_point; i < indices.size(); ++i) test_indices.push_back(indices[i]);
        }
        
        // Trộn lại lần cuối để các lớp không bị xếp liền nhau
        shuffle(train_indices.begin(), train_indices.end(), g);
        shuffle(test_indices.begin(), test_indices.end(), g);

    } else {
        // --- CÁCH B: CHIA NGẪU NHIÊN THÔNG THƯỜNG (SIMPLE SPLIT) ---
        vector<int> all_indices(total_rows);
        for(size_t i=0; i<total_rows; ++i) all_indices[i] = i;
        
        shuffle(all_indices.begin(), all_indices.end(), g);

        size_t n_test = (total_rows * test_size) / 100;
        size_t split_point = total_rows - n_test;

        for (size_t i = 0; i < split_point; ++i) train_indices.push_back(all_indices[i]);
        for (size_t i = split_point; i < total_rows; ++i) test_indices.push_back(all_indices[i]);
    }

    // BƯỚC 3: Đóng gói kết quả thành 4 Dataframe (X_train, X_test, y_train, y_test)
    vector<Dataframe> result;
    result.push_back(create_subset(X, train_indices)); // [0] X_train
    result.push_back(create_subset(X, test_indices));  // [1] X_test
    result.push_back(create_subset(y, train_indices)); // [2] y_train
    result.push_back(create_subset(y, test_indices));  // [3] y_test

    return result;
}