#include "../include/Train_test_split.h"
#include <algorithm> // shuffle
#include <random>    // mt19937, random_device
#include <iostream>
#include <variant>   // visit

// --- Triển khai các hàm Private ---

string Train_test_split::variant_to_string(const DataType& val) {
    return visit([](auto&& arg) -> string {
        using T = decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, string>) return arg;
        else return to_string(arg);
    }, val);
}

Dataframe Train_test_split::create_subset(const Dataframe& original, const vector<int>& indices) {
    Dataframe sub;
    sub.cols = original.cols;
    sub.nCols = original.nCols;
    sub.nRows = indices.size();

    // Khởi tạo vector rỗng cho các cột (tối ưu bộ nhớ)
    for (const string& col_name : sub.cols) {
        sub.data[col_name].reserve(indices.size());
    }

    // Copy dữ liệu
    for (const string& col_name : sub.cols) {
        const auto& src_vec = original.data.at(col_name);
        auto& dest_vec = sub.data[col_name];
        for (int idx : indices) {
            dest_vec.push_back(src_vec[idx]);
        }
    }
    return sub;
}

// --- Triển khai hàm Public ---

// LƯU Ý: Ở file .cpp KHÔNG được viết lại "= true", chỉ viết tên biến
vector<Dataframe> Train_test_split::train_test_split(const Dataframe& X, const Dataframe& y, int test_size, bool stratify) {
    if (X.nRows != y.nRows) {
        cout << "Loi: So luong hang cua X va y khong khop!" << endl;
        return {};
    }

    size_t total_rows = X.nRows;
    vector<int> train_indices;
    vector<int> test_indices;
    
    // Seed cố định để dễ debug (thay bằng random_device nếu muốn ngẫu nhiên thật)
    mt19937 g(42); 

    if (stratify && y.nCols > 0) {
        // --- LOGIC PHÂN TẦNG ---
        map<string, vector<int>> class_groups;
        string label_col = y.cols[0];
        const auto& label_data = y.data.at(label_col);

        for (size_t i = 0; i < total_rows; ++i) {
            string key = variant_to_string(label_data[i]);
            class_groups[key].push_back(i);
        }

        for (auto& [key, indices] : class_groups) {
            shuffle(indices.begin(), indices.end(), g);
            
            size_t n_test_local = (indices.size() * test_size) / 100;
            if (n_test_local == 0 && indices.size() > 1 && test_size > 0) n_test_local = 1;

            size_t split_point = indices.size() - n_test_local;

            for (size_t i = 0; i < split_point; ++i) train_indices.push_back(indices[i]);
            for (size_t i = split_point; i < indices.size(); ++i) test_indices.push_back(indices[i]);
        }
        
        // Trộn lại lần cuối
        shuffle(train_indices.begin(), train_indices.end(), g);
        shuffle(test_indices.begin(), test_indices.end(), g);

    } else {
        // --- LOGIC NGẪU NHIÊN THƯỜNG ---
        vector<int> all_indices(total_rows);
        for(size_t i=0; i<total_rows; ++i) all_indices[i] = i;
        
        shuffle(all_indices.begin(), all_indices.end(), g);

        size_t n_test = (total_rows * test_size) / 100;
        size_t split_point = total_rows - n_test;

        for (size_t i = 0; i < split_point; ++i) train_indices.push_back(all_indices[i]);
        for (size_t i = split_point; i < total_rows; ++i) test_indices.push_back(all_indices[i]);
    }

    vector<Dataframe> result;
    result.push_back(create_subset(X, train_indices));
    result.push_back(create_subset(X, test_indices));
    result.push_back(create_subset(y, train_indices));
    result.push_back(create_subset(y, test_indices));

    return result;
}