#include <iomanip>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "MINST_Loader.h"
#include "Layer.h"
#include "Arena.h"
#include "Network.h"

int main() {
    std::vector<size_t> sizes = {3, 16, 2};
    Network net(sizes, ActivationType::ReLU, ActivationType::Softmax);

    // przygotuj wejście (rozmiar 3)
    float input[3] = {0.2f, 0.7f, 0.1f};

    // target: klasa 1 (one-hot dla rozmiaru 2)
    float target[2] = {0.0f, 1.0f};

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
    float lr = 0.5f;
    net.update(lr, 1);

    // forward po update
    net.forward_pass(input);
    out = net.output_ptr();
    std::cout << "Output (after): ";
    for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
    std::cout << std::endl;

    loss = 0.0;
    for (size_t i = 0; i < net.output_size(); ++i) {
        loss -= target[i] * std::log(std::max(1e-8f, out[i]));
    }
    std::cout << "Loss (after): " << loss << std::endl;

    // statystyki areny (opcjonalnie)
    net.print_stats();

    return 0;
}