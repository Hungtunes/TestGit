/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Thực thi lớp Metrics đánh giá hiệu suất mô hình
*/

#include "../include/Metrics.h"

double Metrics::accuracy_score(const Dataframe& y_true, const vector<string>& y_pred) {
    string label_col = y_true.cols[0];
    const auto& true_data = y_true.data.at(label_col);
    int correct = 0;
    for (size_t i = 0; i < y_true.nRows; ++i) {
        string true_val = visit([](auto&& arg) -> string {
            using T = decay_t<decltype(arg)>;
            if constexpr (is_same_v<T, string>) return arg;
            else return to_string(arg);
        }, true_data[i]);
        if (true_val == y_pred[i]) correct++;
    }
    return (double)correct / y_true.nRows;
}

vector<vector<int>> Metrics::get_confusion_matrix(const Dataframe& y_true, const vector<string>& y_pred, vector<string>& labels_out) {
    set<string> unique_labels;
    string label_col = y_true.cols[0];
    const auto& true_data_vec = y_true.data.at(label_col); // [Optimize] Cache vector reference

    // 1. Lấy tất cả các nhãn
    for (size_t i = 0; i < y_true.nRows; ++i) {
        string s = visit([](auto&& arg) -> string {
            using T = decay_t<decltype(arg)>;
            if constexpr (is_same_v<T, string>) return arg;
            else return to_string(arg);
        }, true_data_vec[i]);
        unique_labels.insert(s);
    }
    for (const string& s : y_pred) unique_labels.insert(s);

    labels_out.assign(unique_labels.begin(), unique_labels.end());
    map<string, int> label_to_index;
    for (size_t i = 0; i < labels_out.size(); ++i) label_to_index[labels_out[i]] = i;

    size_t n = labels_out.size();
    vector<vector<int>> matrix(n, vector<int>(n, 0));

    // 2. Điền ma trận (Dùng vector cached để nhanh hơn)
    for (size_t i = 0; i < y_true.nRows; ++i) {
        string true_val = visit([](auto&& arg) -> string {
            using T = decay_t<decltype(arg)>;
            if constexpr (is_same_v<T, string>) return arg;
            else return to_string(arg);
        }, true_data_vec[i]);
        
        matrix[label_to_index[true_val]][label_to_index[y_pred[i]]]++;
    }
    return matrix;
}

void Metrics::print_confusion_matrix(const Dataframe& y_true, const vector<string>& y_pred) {
    vector<string> labels;
    vector<vector<int>> cm = get_confusion_matrix(y_true, y_pred, labels);
    int w = 15; // Width
    cout << "\n========== CONFUSION MATRIX ==========\n";
    cout << setw(w) << " ";
    for (const string& l : labels) cout << setw(w) << l;
    cout << "\n" << setw(w) << " " << string(labels.size() * w, '-') << endl;
    for (size_t i = 0; i < cm.size(); ++i) {
        cout << setw(w) << labels[i] << "|";
        for (int val : cm[i]) cout << setw(w) << val;
        cout << endl;
    }
    cout << "======================================\n";
}

void Metrics::classification_report(const Dataframe& y_true, const vector<string>& y_pred) {
    vector<string> labels;
    vector<vector<int>> cm = get_confusion_matrix(y_true, y_pred, labels);
    
    cout << "\n========== CLASSIFICATION REPORT ==========\n";
    cout << setw(20) << "Class" << setw(12) << "Precision" << setw(12) << "Recall" << setw(12) << "F1-Score" << endl;
    cout << "--------------------------------------------------------" << endl;

    double m_pre = 0, m_rec = 0, m_f1 = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        int tp = cm[i][i];
        int fp = 0, fn = 0;
        for (size_t j = 0; j < labels.size(); ++j) {
            if (i != j) {
                fp += cm[j][i]; // Cột i (Predicted) nhưng Hàng j (Actual) -> False Positive
                fn += cm[i][j]; // Hàng i (Actual) nhưng Cột j (Predicted) -> False Negative
            }
        }
        double p = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
        double r = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
        double f1 = (p + r) > 0 ? 2 * p * r / (p + r) : 0;
        
        m_pre += p; m_rec += r; m_f1 += f1;
        cout << setw(20) << labels[i] << setw(12) << fixed << setprecision(2) << p 
             << setw(12) << r << setw(12) << f1 << endl;
    }
    size_t n = labels.size();
    if (n > 0) {
        cout << "--------------------------------------------------------" << endl;
        cout << setw(20) << "Macro Avg" << setw(12) << m_pre/n << setw(12) << m_rec/n << setw(12) << m_f1/n << endl;
    }
    cout << "Accuracy: " << accuracy_score(y_true, y_pred) << endl;
}