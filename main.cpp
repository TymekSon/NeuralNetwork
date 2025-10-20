#include <iomanip>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "MINST_Loader.h"
#include "Layer.h"
#include "Arena.h"
#include "Network.h"

std::vector<std::pair<float, float>> generate_input(int len) {
    std::vector<std::pair<float, float>> x(len+1);
    int count = 0;
    for (size_t i = 0; i < len; ++i) {
        x[i].first = std::rand()%2;
        if (x[i].first == 1) {
            count ++;
        }
        x[i].second = 0;
    }
    x[count].second = 1;
    return x;
}



int main() {
    std::vector<size_t> sizes = {3, 8, 7};
    Network net(sizes, ActivationType::ReLU, ActivationType::Softmax);

    // przygotuj wejście (rozmiar 3)
    float input[3] = {0.6f, 0.2f, 0.8f};

    // target: klasa 1 (one-hot dla rozmiaru 2)
    float target[7] = {0.0f, 0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f};

    float lr = 0.4f;

    for (size_t i = 0; i < 20; ++i) {
        // przed treningiem -> forward
        net.forward_pass(input);
        float* out = net.output_ptr();
        std::cout << "Output (before): ";
        for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
        std::cout << std::endl;

        // policz strata cross-entropy: -sum y*log(a)
        double loss = 0.0;
        for (size_t i = 0; i < net.output_size(); ++i) {
            loss -= target[i] * std::log(std::max(1e-8f, out[i]));
        }
        std::cout << "Loss (before): " << loss << std::endl;

        // backward (one sample)
        net.backward_pass(target);

        // update (proste SGD)
        net.update(lr, 1);
    }


    // // forward po update
    // net.forward_pass(input);
    // out = net.output_ptr();
    // std::cout << "Output (after): ";
    // for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
    // std::cout << std::endl;
    //
    // loss = 0.0;
    // for (size_t i = 0; i < net.output_size(); ++i) {
    //     loss -= target[i] * std::log(std::max(1e-8f, out[i]));
    // }
    // std::cout << "Loss (after): " << loss << std::endl;

    // statystyki areny (opcjonalnie)
    net.print_stats();

    return 0;
}