//
// Created by chomi on 16.07.2025.
//
#include <cstddef>

#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H

struct LayerConfig {
    size_t input_size;    // liczba wejść do warstwy
    size_t output_size;   // liczba neuronów w warstwie

    // Wskaźniki na bufor pamięci zaalokowany przez MemoryArena
    float* weights_ptr;   // in_size * out_size elementów
    float* biases_ptr;    // out_size elementów
    float* z_ptr;         // out_size elementów (surowe sumy)
    float* a_ptr;         // out_size elementów (aktywacje)
    float* delta_ptr;     // out_size elementów (delta)
    float* grad_w_ptr;    // in_size * out_size elementów (gradienty wag)
    float* grad_b_ptr;    // out_size elementów (gradienty biasów)
};

#endif //LAYER_CONFIG_H
