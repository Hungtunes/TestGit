/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Định nghĩa về lớp DataFrame: Lưu trữ dữ liệu từ file csv vào trong chương trình
*/

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>

using namespace std;

// Định nghĩa các kiểu dữ liệu có thể chứa trong một Frame: int, double, string
using DataType = variant<int, double, string>;


class Dataframe {
public:
    
    // Định nghĩa cách lưu trữ một DataFame
    vector<string> cols; // Các thuộc tính
    unordered_map<string, vector<DataType>> data;

    // Số hàng và số cột
    size_t nRows = 0;
    size_t nCols = 0;

    
    // Phương thức đọc file csv vào trong DataFrame
    bool readCSV(string filename);

    // Phương thức hiển thị các thông tin tổng quan của dữ liệu vừa đọc được
    void info();

    // Phương thức giúp hiển thị trực quan n dòng đầu của dữ liệu
    void head(int n = 5);

    // Phương thức giúp tách cột nhãn Label ra khỏi Dataframe
    Dataframe label(string label_name);
};
