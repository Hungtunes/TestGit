#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <cmath>
#include "Dataframe.h"

using namespace std;

class GaussianNB {
private:
    // Các nhãn
    vector<string> classes;
    // Xác suất tiên nghiệm
    map<string, double> priors;
    // Lưu trung bình, phương sai
    map<string, unordered_map<string, double>> mean_map;
    map<string, unordered_map<string, double>> var_map;
    // Hàm tính xác suất dựa trên phân phối chuẩn
    double calculate_probability(double x, double mean, double var);
    
public:

    // Train mô hình
    void fit(const Dataframe& X, const Dataframe& y);
    // Dự đoán
    vector<string> predict(const Dataframe& X);
};