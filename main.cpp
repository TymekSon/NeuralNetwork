#include <iomanip>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>

#include "MINST_Loader.h"
#include "DenseLayer.h"
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
    MINST_Loader loader;

    std::string train_imgs_f = "../Data/trainImages.idx3-ubyte";
    std::string train_labels_f = "../Data/trainLabels.idx1-ubyte";

    std::string test_imgs_f = "../Data/testImages.idx3-ubyte";
    std::string test_labels_f = "../Data/testLabels.idx1-ubyte";

    auto train_imgs = loader.load_MINST_Images(train_imgs_f);
    auto train_labels = loader.load_MINST_Labels(train_labels_f);

    auto norm_train_imgs = loader.normalize_MINST_Images(train_imgs);
    auto norm_train_labels = loader.to_one_hot(train_labels, 10);

    auto test_imgs = loader.load_MINST_Images(test_imgs_f);
    auto test_labels = loader.load_MINST_Labels(test_labels_f);

    auto norm_test_imgs = loader.normalize_MINST_Images(test_imgs);
    auto norm_test_labels = loader.to_one_hot(test_labels, 10);

    auto layer1 = std::make_unique<DenseLayer>(784, 128, ActivationType::ReLU);
    auto layer2 = std::make_unique<DenseLayer>(128, 10, ActivationType::Softmax);

    std::vector<std::unique_ptr<BaseLayer>> layers;
    layers.push_back(std::move(layer1));
    layers.push_back(std::move(layer2));

    Network net(std::move(layers), 784);

    int batch_size = 16;
    float lr = 0.05f;
    float lr_decay = 0.0002f;

    std::cout << "Learning..." << std::endl;

    for (size_t epoch = 0; epoch < 2; ++epoch) {
        std::vector<size_t> indices(norm_test_imgs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));

        for (size_t i = 0; i < norm_test_imgs.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, norm_test_imgs.size());

            for (size_t j = i; j < end; ++j) {
                float* input = norm_test_imgs[indices[j]].data();
                float* target = norm_test_labels[indices[j]].data();

                net.forward_pass(input);
                net.backward_pass(target);
            }

            net.update(lr-lr_decay*epoch, 0.8, end - i); // jedna aktualizacja po batchu
        }
    }

    std::cout << "Testing accuracy..." << std::endl;

    int correct = 0;

    for (size_t i = 0; i < test_imgs.size(); ++i) {
        float* input = norm_train_imgs[i].data();
        float* target = norm_train_labels[i].data();

        net.forward_pass(input);
        float* out = net.output_ptr();

        int pred = std::distance(out, std::max_element(out, out + net.output_size()));
        int actual = std::distance(target, std::max_element(target, target + net.output_size()));

        if (pred == actual) correct++;
    }

    std::cout << "Accuracy: " << (float)correct / test_imgs.size() * 100.0f << " %" << std::endl;

    // int steps = 20000;
    // for (size_t i = 0; i < steps; ++i) {
    //     auto data = train_data(9);
    //     float* input = data.first;
    //     float* target = data.second;
    //
    //     // przed treningiem -> forward
    //     net.forward_pass(input);
    //     float* out = net.output_ptr();
    //     //std::cout << "Output (before): ";
    //     //for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
    //     //std::cout << std::endl;
    //
    //     // policz strata cross-entropy: -sum y*log(a)
    //     double loss = 0.0;
    //     for (size_t i = 0; i < net.output_size(); ++i) {
    //         loss -= target[i] * std::log(std::max(1e-8f, out[i]));
    //     }
    //     //std::cout << "Loss (before): " << loss << std::endl;
    //
    //     // backward (one sample)
    //     net.backward_pass(target);
    //
    //     // update (proste SGD)
    //     net.update(lr, 1);
    //
    //     delete[] input;
    //     delete[] target;
    // }
    //
    // float* test_input = new float[9]{1,0,1,0,1,0,0,0,1};
    //
    // int actual = 0;
    // for (int i = 0; i < 9; ++i) {
    //     if (test_input[i] == 1) actual++;
    // }
    //
    // // SIMPLE TEST
    // net.forward_pass(test_input);
    // float* out = net.output_ptr();
    // int pred =  getPrediction(out, 10);
    // std::cout << "Counting ones, " << actual <<" ones in the array\n";
    // std::cout << "TEST Output (before): ";
    // for (size_t i = 0; i < net.output_size(); ++i) std::cout << out[i] << " ";
    // std::cout << std::endl;
    // std::cout << "Prediction: " << pred <<"\n\n";
    //
    // net.print_stats();
    //
    // delete[] test_input;
    //
    // // SCIENTIFIC TEST
    // int count = 0;
    // int test_steps = 100000;
    // for (int i = 0; i < test_steps; ++i) {
    //     auto test_data = train_data(9);
    //     float* test = test_data.first;
    //     float* label = test_data.second;
    //
    //     net.forward_pass(test);
    //     float* out = net.output_ptr();
    //     int pred =  getPrediction(out, 10);
    //
    //     int actual = std::distance(label, std::max_element(label, label + 10));
    //
    //     if (pred == actual) count ++;
    //
    //     delete[] test;
    //     delete[] label;
    // }
    // std::cout << "Accuracy: " << (float)100*count/test_steps << " %" << std::endl;

    return 0;
}