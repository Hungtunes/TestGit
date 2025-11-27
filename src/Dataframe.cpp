#include "../include/Dataframe.h" // Import header vừa tạo
#include <fstream>
#include <sstream>
#include <algorithm>


bool Dataframe::readCSV(string filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    string line;
    getline(file, line);
    stringstream ss(line);
    string col;

    while (getline(ss, col, ',')) {
        // Xoa ky tu thua neu co
        col.erase(remove(col.begin(), col.end(), '\r'), col.end());
        col.erase(remove(col.begin(), col.end(), '\n'), col.end());
        cols.push_back(col);
        data[col] = {};
    }
    nCols = data.size();

    while (getline(file, line)) {
        stringstream ss2(line);
        string val;
        int idx = 0;
        while (getline(ss2, val, ',')) {
            val.erase(remove(val.begin(), val.end(), '\r'), val.end());
            val.erase(remove(val.begin(), val.end(), '\n'), val.end());
            
            // Cot cuoi cung la Label (string), cac cot truoc la feature (double)
            if (idx != nCols - 1) {
                try {
                    data[cols[idx]].push_back(stod(val));
                } catch (...) {
                    data[cols[idx]].push_back(0.0); // Fallback neu loi
                }
            } else {
                data[cols[idx]].push_back(val);
            }
            idx++;
        }
        nRows++;
    }
    return true;
}

void Dataframe::info() {
    cout << endl << "----Dataframe info----" << endl;
    cout << "Row: " << nRows << endl;
    cout << "Col: " << nCols << endl;
    for (const auto &col : cols) {
        cout << col << endl;
    }
}

void Dataframe::head(int n) {
    cout << "----Mot so gia tri cua dac trung trong DataFrame---- \n";
    for (const auto &col : cols) {
        cout << col << "\t";
        for (int i = 0; i < n; i++) {
            visit([](auto&& value) {
                cout << value << "\t";
            }, data[col].at(i));
        }
        cout << endl;
    }
}

Dataframe Dataframe::label(string label_name) {
    Dataframe df_label;
    
    // 1. Kiem tra cot co ton tai khong
    auto it = data.find(label_name);
    if (it == data.end()) {
        cout << "Loi: Khong tim thay cot label!" << endl;
        return df_label;
    }

    // 2. Setup thong so cho df_label
    df_label.nRows = nRows;
    df_label.nCols = 1;
    df_label.cols.push_back(label_name);
    df_label.data[label_name] = move(it->second); 

    // 3. Xoa label -> features
    data.erase(it); 
    cols.erase(remove(cols.begin(), cols.end(), label_name), cols.end());
    nCols--;

    return df_label;
}