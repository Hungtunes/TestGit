/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Định nghĩa về lớp StandardScaller: chuẩn hóa giá trị của các thuộc tính về dạng phân phối chuẩn
*/

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include "Dataframe.h" 

using namespace std;

class StandardScaler {
public:
    // Lưu trữ thông tin độ lệch chuẩn và trung bình
    unordered_map<string, double> mean_map;
    unordered_map<string, double> std_map;

    // Phương thức tính độ lệch chuẩn và trung bình cho từng thuộc tính
    void fit(const Dataframe& X);

    // Phương thức chuẩn hóa dữ liệu
    void transform(Dataframe& X);

    // Phương thức tính toán và chuẩn hóa dữ 
    void fit_transform(Dataframe& X);

private:
    // Hàm tiện ích: Chuyển đổi variant sang double
    double get_value_as_double(const DataType& val);
};