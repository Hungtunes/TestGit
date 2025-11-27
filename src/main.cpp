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
    // string filePath = "C:\\Users\\teo\\Downloads\\data_test.csv";
    string filePath = "C:\\Users\\teo\\Downloads\\DataFull.csv";
    cout << "Dang tien hanh doc file csv ..." << endl;
    if (df.readCSV(filePath)) {
        cout << "Da doc file thanh cong!" << endl;
    } else {
        cout << "Duong dan khong hop le" << endl;
        return 1;
    }

    clock_t end = clock();
    double time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "\nThoi gian doc file csv: " << time_taken << " s\n";
    // 2. Tach rieng features va label
    start = clock();
    Dataframe y = df.label(" Label");
    Dataframe X = move(df);    
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "\nThoi gian tach features/label: " << time_taken << " s\n";

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
    cout << "\nThoi gian tach train, test: " << time_taken << " s\n";
    // 4. Chuan hoa feature 
    start = clock();
    cout << endl << "Tien hanh chuan hoa ..." << endl;
    StandardScaler scl;
    //fit tren tap train
    scl.fit(X_train);
    scl.fit_transform(X_train);
    scl.fit_transform(X_test);
    end = clock();
    time_taken = double(end - start) / CLOCKS_PER_SEC;
    cout << "\nThoi gian chuan hoa: " << time_taken << " s\n";

    return 0;
}