/*
.__                          __                              
|  |__  __ __  ____    _____/  |_ __ __  ____   ____   ______
|  |  \|  |  \/    \  / ___\   __\  |  \/    \_/ __ \ /  ___/
|   Y  \  |  /   |  \/ /_/  >  | |  |  /   |  \  ___/ \___ \ 
|___|  /____/|___|  /\___  /|__| |____/|___|  /\___  >____  >
     \/           \//_____/                 \/     \/     \/ 
*/

/*
- Hàm main của chương trình (gợi ý)
*/

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <string>
#include <algorithm>
#include <ctime>
#include <limits>
#include "../include/Dataframe.h"
#include "../include/Train_test_split.h"
#include "../include/Naivebayes.h"
#include "../include/Metrics.h"
#include "../include/StandardScaler.h"

using namespace std;

int main() {

    string filePath;
    string labelName;
    double testSize;

    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    #endif

    // 0. Nhập thông tin
    cout << R"(
     _   _                             ____                           
    | \ | |  __ _ (_)__   __  ___     | __ )   __ _  _   _   ___  ___ 
    |  \| | / _` || |\ \ / / / _ \    |  _ \  / _` || | | | / _ \/ __|
    | |\  || (_| || | \ V / |  __/    | |_) || (_| || |_| ||  __/\__ \
    |_| \_| \__,_||_|  \_/   \___|    |____/  \__,_| \__, | \___||___/
                                                     |___/            
     ____   _                   _   __  _             
    / ___| | |  __ _  ___  ___ (_) / _|(_)  ___  _ __ 
    | |    | | / _` |/ __|/ __|| || |_ | | / _ \| '__|
    | |___ | || (_| |\__ \\__ \| ||  _|| ||  __/| |   
    \____| |_| \__,_||___/|___/| || |  |_| \___||_|   
                               |_|                )" << endl;
    cout << "====================== Thiết lập ======================" << endl;
    cout << "⚠️ Lưu ý:" << endl; 
    cout << "   ◆ Đường dẫn phải có dạng như ví dụ: E:/Code/NaiveBayes/Dataset/datafinal.csv" << endl;
    cout << "   ◆ Cột Label nằm ở cuối cùng" << endl;
    cout << "   ◆ Dữ liệu cần được tiền xử lý trước: Giá trị các thuộc tính đều là số nguyên, không có giá trị NULL" << endl;
    cout << "   ◆ Tỷ lệ tập Test ví dụ: 30 cho 30% là trong tập Test" << endl;
    cout << "1. Nhập đường dẫn file CSV: ";
    getline(cin, filePath);

    cout << "2. Nhập tên cột nhãn (Label): ";
    getline(cin, labelName);

    cout << "3. Nhập tỷ lệ tập Test : ";
    cin >> testSize;
    cout << "=======================================================" << endl << endl;


    // 1. Đọc dữ liệu
    clock_t start = clock();
    Dataframe df;
    
    if (!df.readCSV(filePath)) {
        return 404;
    }

    clock_t end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "⏳ Thời gian đọc dữ liệu: " << time_taken << "s" << endl << endl;


    // 2. Tách thuộc tính và nhãn
    start = clock();
    Dataframe y = df.label(labelName); 
    Dataframe X = move(df);    
    
    end = clock();
    cout << "⏳ Thời gian tách features/label: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl << endl;


    // 3. Tách tập Train/Test
    start = clock();
    cout << "🚀 Tiến hành tách train test (Ratio: " << testSize << "%) ..." << endl;
    
    Train_test_split splitter;
    vector<Dataframe> result = splitter.train_test_split(X, y, testSize, true);

    Dataframe X_train = result[0];
    Dataframe X_test = result[1];
    Dataframe y_train = result[2];
    Dataframe y_test = result[3];

    cout << "   ◆ Số bản ghi tập Train: " << X_train.nRows << endl;
    cout << "   ◆ Số bản ghi tập Test: " << X_test.nRows << endl;

    end = clock();
    cout << "⏳ Thời gian tách train/test: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl << endl;


    // 4. Chuẩn hóa
    start = clock();
    cout << "🚀 Tiến hành chuẩn hóa dữ liệu..." << endl;
    StandardScaler scl;
    scl.fit(X_train);
    scl.fit_transform(X_train);
    scl.transform(X_test);
    
    end = clock();
    cout << "⏳ Thời gian chuẩn hóa: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl << endl;


    // --- 5. Huấn luyện Model GaussianNB
    start = clock();
    cout << "🚀 Tiến hành huấn luyện mô hình..." << endl;
    GaussianNB model;
    model.fit(X_train, y_train);
    
    end = clock();
    cout << "⏳ Thời gian huấn luyện: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl << endl;


    // 6. Dự đoán
    start = clock();
    cout << "🚀 Tiến hành dự đoán ...";
    vector<string> y_pred = model.predict(X_test);
    cout << endl;
    end = clock();
    cout << "⏳ Thời gian dự đoán: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl << endl;

    // --- 7. ĐÁNH GIÁ MÔ HÌNH ---
    int choice = -1;

    while (true) {
        cout << "\n=================== KẾT QUẢ ĐÁNH GIÁ ===================\n";
        cout << "1. Xem báo cáo phân loại (Classification Report)\n";
        cout << "2. Xem ma trận nhầm lẫn (Confusion Matrix / Metrics)\n";
        cout << "3. Xem cả hai\n";
        cout << "4. Thoát chương trình\n";
        cout << "👉 Lựa chọn của bạn: ";

        if (!(cin >> choice)) {
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cout << "\n❌ Vui lòng nhập số từ 1 đến 4!\n";
            continue;
        }

        if (choice == 4) {
            cout << "\n👋 Đang thoát chương trình...\n";
            break;
        }

        cout << "\n--------------------------------------------------------\n";

        switch (choice) {
            case 1:
                cout << "📍 Đang hiển thị: Classification Report...\n";
                Metrics::classification_report(y_test, y_pred);
                break;

            case 2:
                cout << "📍 Đang hiển thị: Confusion Matrix...\n";
                Metrics::print_confusion_matrix(y_test, y_pred);
                break;

            case 3:
                cout << "📍 Đang hiển thị: Báo cáo tổng hợp...\n\n";
                Metrics::classification_report(y_test, y_pred);
                cout << '\n';
                Metrics::print_confusion_matrix(y_test, y_pred);
                break;

            default:
                cout << "❌ Lựa chọn không hợp lệ! Hãy nhập từ 1 đến 4.\n";
                break;
        }
    }

    cout << "========================================================\n";

    return 0;
}