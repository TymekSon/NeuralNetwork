//
// Created by chomi on 16.07.2025.
//

#include "softmax_cross_entropy.h"
#include <algorithm>
#include <cmath>

float softmax_cross_entropy::forward(const float* z, const float* y, float* p, size_t len){
    float z_max = z[0];
    for (size_t i = 1; i < len; ++i) z_max = std::max(z_max, z[i]);

    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i){
      p[i] = std::exp(z[i] - z_max);
      sum += p[i];
    };

    float loss = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        p[i] /= sum;
        if (y[i] > 0.5f) {  // y jako one-hot (dokładnie 1.0f)
            loss -= std::log(std::max(p[i], 1e-8f));
        }
    }
    return loss;
}

void softmax_cross_entropy::backward(const float* p, const float* y, float* grad_z, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        grad_z[i] = p[i] - y[i];
    }
}