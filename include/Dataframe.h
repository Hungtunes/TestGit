#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>

using namespace std;

using DataType = variant<int, double, string>;

class Dataframe {
public:
    
    unordered_map<string, vector<DataType>> data;
    size_t nRows = 0;
    size_t nCols = 0;
    vector<string> cols;
    
    // Đọc file CSV
    bool readCSV(string filename);

    // Hiển thị thông tin tổng quan
    void info();

    // Hiển thị n dòng đầu
    void head(int n = 5);

    // Tách cột label
    Dataframe label(string label_name);
};
