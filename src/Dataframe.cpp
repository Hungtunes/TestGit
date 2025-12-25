/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Thực thi Lớp DataFrame đã được định nghĩa
*/


#include "../include/Dataframe.h"
#include <fstream>
#include <sstream>
#include <algorithm>

// Phương thức đọc file csv vào trong DataFrame
bool Dataframe::readCSV(string filename) {
    // Mở file csv cần đọc có path là filename
    cout << "🚀 Đang tiến hành đọc dữ liệu ..." << endl;
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "❌ Đọc file thất bại. Vui lòng kiểm tra lại File path!" << endl;
        return false;
    }

    string line;
    getline(file, line);
    stringstream ss(line);
    string col;

    // Đọc lần lượt các thuộc tính
    while (getline(ss, col, ',')) {
        col.erase(remove(col.begin(), col.end(), '\r'), col.end());
        col.erase(remove(col.begin(), col.end(), '\n'), col.end());
        cols.push_back(col);
        data[col] = {};
    }
    nCols = data.size();

    // Đọc lần lượt từng giá trị của mỗi thuộc tính từ trên xuống dưới
    while (getline(file, line)) {
        stringstream ss2(line);
        string val;
        int idx = 0;
        while (getline(ss2, val, ',')) {
            val.erase(remove(val.begin(), val.end(), '\r'), val.end());
            val.erase(remove(val.begin(), val.end(), '\n'), val.end());
            
            // Mặc định cột cuối cùng là nhãn, các cột còn lại là thuộc tính chỉ chứa số
            if (idx != nCols - 1) {
                try {
                    data[cols[idx]].push_back(stod(val));
                } catch (...) {
                    data[cols[idx]].push_back(0.0); // Giá trị mặc định nếu đọc lỗi
                }
            } else {
                data[cols[idx]].push_back(val);
            }
            idx++;
        }
        nRows++;
    }
    cout << "✅ Đọc thành công file CSV vào DataFrame!" << endl;
    return true;
}

// Phương thức hiển thị các thông tin tổng quan của dữ liệu vừa đọc được
void Dataframe::info() {
    cout << endl << "----Thông tin cơ bản của DataFrame đọc được----" << endl;
    cout << "Số hàng: " << nRows << endl;
    cout << "Số cột: " << nCols << endl;
    cout << "Số thuộc tính: " << nCols - 1 << endl;
    cout << "----Chi tiết thông tin về thuộc tính và nhãn----" << endl;
    for (const auto &col : cols) {
        cout << col << endl;
    }
}

// Phương thức giúp hiển thị trực quan n dòng đầu của dữ liệu
void Dataframe::head(int n) {
    cout << "----Một số giá trị của thuộc tính trong DataFrame----" << endl;
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

// Phương thức giúp tách cột nhãn Label ra khỏi Dataframe
Dataframe Dataframe::label(string label_name) {
    Dataframe df_label;
    
    // Kiểm tra label_name có chính xác
    auto it = data.find(label_name);
    if (it == data.end()) {
        cout << "Lỗi không tìm thấy cột Label" << endl;
        return df_label;
    }

    // Setup thông số cho df_label
    df_label.nRows = nRows;
    df_label.nCols = 1;
    df_label.cols.push_back(label_name);
    df_label.data[label_name] = move(it->second); 

    // Xóa Label khỏi DataFrame
    data.erase(it); 
    cols.erase(remove(cols.begin(), cols.end(), label_name), cols.end());
    nCols--;

    // Trả về Label kết quả
    cout << "✅ Đã tách thành công Thuộc tính và Nhãn" << endl;
    return df_label;
}