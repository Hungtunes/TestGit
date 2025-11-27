# 1. Phiên bản CMake tối thiểu
cmake_minimum_required(VERSION 3.10)

# 2. Tên dự án
project(NaiveBayesClassifier)

# 3. Yêu cầu C++17 (Vì code bạn dùng std::variant)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 4. Khai báo thư mục chứa file .h (Header)
include_directories(include)

# 5. Khai báo tất cả các file .cpp (Source)
# Bạn có thể liệt kê tay hoặc dùng file(GLOB ...)
set(SOURCES
    src/main.cpp
    src/Dataframe.cpp
    src/Train_test_split.cpp
    src/GaussianNB.cpp
    src/Metrics.cpp
    src/StandardScaler.cpp
)

# 6. Tạo file thực thi (Executable)
# Tên file chạy sẽ là "model"
add_executable(model ${SOURCES})