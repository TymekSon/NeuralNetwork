#include <iomanip>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstring>
#include "MINST_Loader.h"
#include "Layer.h"
#include "Arena.h"
#include "Network.h"

std::pair<float*, float*> train_data(int size) {
    // losowa liczba jedynek od 0 do size
    int num_ones = std::rand() % (size + 1);

    float* train_sample = new float[size]();
    float* train_label = new float[size + 1]();

    if (num_ones == 0) {
        // wszystkie zera — nic nie robimy
    }
    else if (num_ones == size) {
        // wszystkie jedynki
        for (int i = 0; i < size; ++i)
            train_sample[i] = 1;
    }
    else {
        // losowo rozmieść jedynki
        int placed = 0;
        while (placed < num_ones) {
            int idx = std::rand() % size;
            if (train_sample[idx] == 0) {
                train_sample[idx] = 1;
                placed++;
            }
        }
    }
    train_label[num_ones] = 1;

    return {train_sample, train_label};
}

int getPrediction(float* arr, int size) {
    if (size <= 0) return -1; // brak danych

    int maxIndex = 0;
    float maxValue = arr[0];

    for (int i = 1; i < size; i++) {
        if (arr[i] > maxValue) {
            maxValue = arr[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

int main() {
    std::vector<size_t> sizes = {9, 6, 10};
    Network net(sizes, ActivationType::ReLU, ActivationType::Softmax);

    float lr = 0.02f;

    int steps = 100000;
    for (size_t i = 0; i < steps; ++i) {
        auto data = train_data(9);
        float* input = data.first;
        float* target = data.second;

        // przed treningiem -> forward
        net.forward_pass(input);
        float* out = net.output_ptr();
        //std::cout << "Output (before): ";
        //for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
        //std::cout << std::endl;

        // policz strata cross-entropy: -sum y*log(a)
        double loss = 0.0;
        for (size_t i = 0; i < net.output_size(); ++i) {
            loss -= target[i] * std::log(std::max(1e-8f, out[i]));
        }
        //std::cout << "Loss (before): " << loss << std::endl;

        // backward (one sample)
        net.backward_pass(target);

        // update (proste SGD)
        net.update(lr, 1);

        delete[] input;
        delete[] target;
    }

    float* test_input = new float[9]{1,1,1,1,0,0,0,0,0};

    int actual = 0;
    for (int i = 0; i < 9; ++i) {
        if (test_input[i] == 1) actual++;
    }

    // SIMPLE TEST
    net.forward_pass(test_input);
    float* out = net.output_ptr();
    int pred =  getPrediction(out, 10);
    std::cout << "Counting ones, " << actual <<" ones in the array\n";
    std::cout << "TEST Output (before): ";
    for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
    std::cout << std::endl;
    std::cout << "Prediction: " << pred <<"\n\n";

    net.print_stats();

    delete[] test_input;

    // SCIENTIFIC TEST
    int count = 0;
    int test_steps = 1000000;
    for (int i = 0; i < test_steps; ++i) {
        auto test_data = train_data(9);
        float* test = test_data.first;
        float* label = test_data.second;

        net.forward_pass(test);
        float* out = net.output_ptr();
        int pred =  getPrediction(out, 10);

        int actual = std::distance(label, std::max_element(label, label + 10));

        if (pred == actual) count ++;

        delete[] test;
        delete[] label;
    }
    std::cout << "Accuracy: " << (float)100*count/test_steps << " %" << std::endl;

    return 0;
}