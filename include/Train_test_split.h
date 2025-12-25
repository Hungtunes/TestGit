/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Định nghĩa về lớp TrainTestSplit: Chia dữ liệu Train/Test
*/

#pragma once

#include <vector>
#include <string>
#include <map>
#include "Dataframe.h"

using namespace std;

class Train_test_split {
private:
    // Phương thức tiện ích 
    string variant_to_string(const DataType& val);

    // Trích xuất một tập dữ liệu con (Subset) dựa trên danh sách chỉ số 
    Dataframe create_subset(const Dataframe& original, const vector<int>& indices);

public:
    vector<Dataframe> train_test_split(const Dataframe& X, const Dataframe& y, int test_size, bool stratify = true);
};