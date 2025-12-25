/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Định nghĩa về lớp Metrics: Đánh giá mô hình
*/

#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <algorithm>
#include "Dataframe.h"

using namespace std;

class Metrics {
public:
    static double accuracy_score(const Dataframe& y_true, const vector<string>& y_pred);

    static void classification_report(const Dataframe& y_true, const vector<string>& y_pred);

    static vector<vector<int>> get_confusion_matrix(const Dataframe& y_true, const vector<string>& y_pred, vector<string>& labels_out);
    
    static void print_confusion_matrix(const Dataframe& y_true, const vector<string>& y_pred);
};