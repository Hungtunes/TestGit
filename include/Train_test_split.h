#pragma once

#include <vector>
#include <string>
#include <map>
#include "Dataframe.h" // Import để hiểu Dataframe và DataType

using namespace std;

class Train_test_split {
private:
    // Helper: Chuyển đổi variant sang string để phân loại
    string variant_to_string(const DataType& val);

    // Helper: Tạo Dataframe con dựa trên index
    Dataframe create_subset(const Dataframe& original, const vector<int>& indices);

public:
    // Hàm chính
    // LƯU Ý: Tham số mặc định (stratify = true) chỉ khai báo ở Header
    vector<Dataframe> train_test_split(const Dataframe& X, const Dataframe& y, int test_size, bool stratify = true);
};