#include "../include/GaussianNB.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>

const double PI = 3.14159265358979323846;

inline double calculate_log_pdf(double x, double mean, double var) {
    double diff = x - mean;
    return -0.5 * (log(2.0 * PI * var) + (diff * diff) / var);
}

void GaussianNB::fit(const Dataframe& X, const Dataframe& y) {
    classes.clear();
    priors.clear();
    mean_map.clear();
    var_map.clear();

    string label_col = y.cols[0];
    const auto& y_data = y.data.at(label_col);
    size_t n_rows = X.nRows;

    map<string, vector<int>> class_indices;
    for (size_t i = 0; i < n_rows; ++i) {
        string label = visit([](auto&& arg) -> string {
             using T = decay_t<decltype(arg)>;
             if constexpr (is_same_v<T, string>) return arg;
             else return to_string(arg);
        }, y_data[i]);
        class_indices[label].push_back(i);
    }

    double global_max_var = 0.0; 

    for (auto const& [cls, indices] : class_indices) {
        classes.push_back(cls);
        double n_samples_in_class = indices.size();

        priors[cls] = n_samples_in_class / static_cast<double>(n_rows);

        for (const string& col_name : X.cols) {
            double sum = 0;
            const auto& col_data = X.data.at(col_name);
            
            // Tính Mean
            for (int idx : indices) {
                visit([&sum](auto&& arg) {
                    using T = decay_t<decltype(arg)>;
                    if constexpr (!is_same_v<T, string>) sum += static_cast<double>(arg);
                }, col_data[idx]);
            }
            double mean = sum / n_samples_in_class;
            mean_map[cls][col_name] = mean;

            // Tính Variance
            double var_sum = 0;
            for (int idx : indices) {
                visit([&var_sum, mean](auto&& arg) {
                    using T = decay_t<decltype(arg)>;
                    if constexpr (!is_same_v<T, string>) {
                        double val = static_cast<double>(arg);
                        var_sum += (val - mean) * (val - mean);
                    }
                }, col_data[idx]);
            }
            double var = var_sum / n_samples_in_class;
            
            // Cập nhật max variance toàn cục để tính epsilon
            if (var > global_max_var) {
                global_max_var = var;
            }
            
            var_map[cls][col_name] = var;
        }
    }

    double epsilon = 1e-9 * global_max_var; 

    // Cập nhật lại tất cả variance với epsilon
    for (auto const& cls : classes) {
        for (const string& col_name : X.cols) {
            var_map[cls][col_name] += epsilon;
        }
    }
}

vector<string> GaussianNB::predict(const Dataframe& X) {
    size_t n_rows = X.nRows;
    size_t n_classes = classes.size();
    
    vector<vector<double>> scores(n_rows, vector<double>(n_classes, 0.0));

    for (size_t c = 0; c < n_classes; ++c) {
        double log_prior = log(priors[classes[c]]);
        for (size_t i = 0; i < n_rows; ++i) {
            scores[i][c] = log_prior;
        }
    }

    for (size_t c = 0; c < n_classes; ++c) {
        string cls = classes[c];
        
        for (const string& col_name : X.cols) {
            double mean = mean_map[cls][col_name];
            double var = var_map[cls][col_name];
  
            double log_var_term = -0.5 * log(2.0 * PI * var); 
            double inv_two_var = 1.0 / (2.0 * var);

            const auto& col_data = X.data.at(col_name); 

            for (size_t i = 0; i < n_rows; ++i) {
                double x = 0.0;
                visit([&x](auto&& arg) {
                    using T = decay_t<decltype(arg)>;
                    if constexpr (!is_same_v<T, string>) x = static_cast<double>(arg);
                }, col_data[i]);

                double diff = x - mean;
                scores[i][c] += log_var_term - (diff * diff) * inv_two_var;
            }
        }
    }

    vector<string> predictions;
    predictions.reserve(n_rows);

    for (size_t i = 0; i < n_rows; ++i) {
        double max_score = -numeric_limits<double>::infinity();
        int best_idx = 0;
        for (size_t c = 0; c < n_classes; ++c) {
            if (scores[i][c] > max_score) {
                max_score = scores[i][c];
                best_idx = c;
            }
        }
        predictions.push_back(classes[best_idx]);
    }

    return predictions;
}