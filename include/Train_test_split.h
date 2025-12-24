#pragma once

#include <vector>
#include <string>
#include <map>
#include "Dataframe.h" // Import để hiểu Dataframe và DataType

using namespace std;

class Train_test_split {
private:
    string variant_to_string(const DataType& val);

    Dataframe create_subset(const Dataframe& original, const vector<int>& indices);

public:
    vector<Dataframe> train_test_split(const Dataframe& X, const Dataframe& y, int test_size, bool stratify = true);
};