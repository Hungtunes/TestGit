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
    unordered_map<string, double> mean_map;
    unordered_map<string, double> std_map;
    void fit(const Dataframe& X);
    void transform(Dataframe& X);
    void fit_transform(Dataframe& X);

private:
    double get_value_as_double(const DataType& val);
};