#include <iostream>
#include "../include/Dataframe.h"
#include "../include/Train_test_split.h"
#include "../include/GaussianNB.h"
#include "../include/Metrics.h"
#include "../include/StandardScaler.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // 1. Doc du lieu
    clock_t start = clock();
    Dataframe df;
    // string filePath = "C:\\Users\\teo\\Downloads\\datafinal.csv";
    string filePath = "E:\\Code\\NaiveBayes\\Dataset\\datafinal.csv";
    cout << "Dang tien hanh doc file csv ..." << endl;
    if (df.readCSV(filePath)) {
        cout << "Da doc file thanh cong!" << endl;
    } else {
        cout << "Duong dan khong hop le" << endl;
        return 1;
    }

    clock_t end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Thoi gian doc file csv: " << time_taken << " s\n";
    // 2. Tach rieng features va label
    start = clock();
    Dataframe y = df.label(" Label");
    Dataframe X = move(df);    
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Thoi gian tach features/label: " << time_taken << " s\n";

    // 3. Train test split
    start = clock();
    cout << "Tien hanh tach train test ..." << endl;
    Train_test_split splitter;
    vector<Dataframe> result = splitter.train_test_split(X, y, 30, true);

    Dataframe X_train = result[0];
    Dataframe X_test = result[1];
    Dataframe y_train = result[2];
    Dataframe y_test = result[3];

    cout << "Size X_train: " << X_train.nRows << endl;
    cout << "Size X_test: " << X_test.nRows << endl;

    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Thoi gian tach train, test: " << time_taken << " s\n";
    // 4. Chuan hoa feature 
    start = clock();
    cout << endl << "Tien hanh chuan hoa ...";
    StandardScaler scl;
    //fit tren tap train
    scl.fit(X_train);
    scl.fit_transform(X_train);
    scl.fit_transform(X_test);
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Thoi gian chuan hoa: " << time_taken << " s\n";

    // 5. Huấn luyện (Gaussian Naive Bayes)
    start = clock();
    cout << "Training model..." << endl;
    GaussianNB model;
    model.fit(X_train, y_train);
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "Thoi gian huan luyen: " << time_taken << " s\n";

    // 6. Dự đoán
    cout << "Predicting..." << endl;
    vector<string> y_pred = model.predict(X_test);

    // 7. Đánh giá (Metrics & Confusion Matrix)
    Metrics::classification_report(y_test, y_pred);
    Metrics::print_confusion_matrix(y_test, y_pred);

    return 0;
}