 #include <bits/stdc++.h>
using namespace std;

// DataType cua mot Dataframe
using DataType = variant<int, double, string>;

const double PI = 3.14159265358979323846;

// Class Dataframe
class Dataframe {
public:
    // Cau truc Dataframe
    unordered_map<string, vector<DataType>> data;
    size_t nRows = 0;
    size_t nCols = 0;
    vector<string> cols;

    // Ham doc file CSV
    bool readCSV(string filename) {
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

    void info() {
        cout << endl << "----Dataframe info----" << endl;
        cout << "Row: " << nRows << endl;
        cout << "Col: " << nCols << endl;
        for (const auto &col : cols) {
            cout << col << endl;
        }
    }

    void head(int n = 5) {
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

    Dataframe label(string label_name) {
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
};

class StandardScaler {
public:
    unordered_map<string, double> mean_map;
    unordered_map<string, double> std_map;

    void fit(Dataframe &X) {
        mean_map.clear();
        std_map.clear();
        int n = X.nRows;
        for (auto &col : X.data) {
            auto &feature = col.second;
            double mean = 0;
            double std = 0;
            double val;
            for (int i = 0; i < n; i++) {
                val = get<double>(feature.at(i));
                mean += val;
            }

            mean = mean/n;

            for (int i = 0; i < n; i++) {
                val = get<double>(feature.at(i));
                std += (val-mean)*(val - mean);
            }
            std = sqrt(std/n);

            mean_map[col.first] = mean;
            std_map[col.first] = std;
        }
    }

    void fit_transform(Dataframe &X) {
        fit(X);
        int n = X.nRows;
        for (auto &col : X.data) {
            auto &feature = col.second;
            for (int i = 0; i < n; i++) {
                feature.at(i) = (get<double>(feature.at(i)) - mean_map[col.first])/std_map[col.first];
            }
        }
    }
};

class Train_test_Split {

};

class GaussianNB {
    // sẽ code tiếp
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // 1. Doc du lieu
    clock_t start = clock();

    Dataframe df;
    string filePath = "C:\\Users\\teo\\Downloads\\data_test.csv";
    cout << "Dang tien hanh doc file csv ..." << endl;
    if (df.readCSV(filePath)) {
        cout << "Da doc file thanh cong!" << endl;
    } else {
        cout << "Duong dan khong hop le" << endl;
        return 1;
    }
    // df.info();
    // df.head();

    clock_t end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "\nThoi gian doc file csv: " << time_taken << " s\n";

    // 2. Tach rieng features va label
    start = clock();
    Dataframe y = df.label(" Label");
    Dataframe X = std::move(df);    
    // cout << "----Label----\n";
    // y.head();
    // cout << "----Features----\n";
    // X.head();
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "\nThoi gian tach features/label: " << time_taken << " s\n";
    // 3. Chuan hoa feature (thu nghiem)
    start = clock();
    StandardScaler scl;
    scl.fit_transform(X);
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "\nThoi gian chuan hoa: " << time_taken << " s\n";
    X.head();
    return 0;
}