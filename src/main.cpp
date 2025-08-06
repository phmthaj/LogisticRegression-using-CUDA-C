#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "logistic.h"

int main() {
    const int n_samples = 5000;
    const int n_features = 20;

    std::vector<float> X(n_samples * n_features);
    std::vector<float> y(n_samples);
    std::vector<float> beta(n_features, 0.0f);

    std::ifstream infile("data.txt");
    if (!infile.is_open()) {
        return 1;
    }

    std::string line;
    int row = 0;
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        float value;
        int count = 0;

        for (int i = 0; i < n_features; ++i) {
            if (!(ss >> value)) {
                return 1;
            }
            X[row * n_features + i] = value;
            count++;
        }

        if (!(ss >> value)) {
            return 1;
        }
        y[row] = value;
        count++;

        if (ss >> value) {
        }

        ++row;
        if (row >= n_samples) break;
    }

    

    infile.close();

    solve(X.data(), y.data(), beta.data(), n_samples, n_features);

    std::cout << "Weight after trained:\n";
    for (int i = 0; i < n_features; ++i) {
        std::cout << beta[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

