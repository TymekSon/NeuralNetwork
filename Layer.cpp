//
// Created by chomi on 08.07.2025.
//

#include "Layer.h"
#include <cmath>
#include <random>
#include "Arena.h"

static float rand_uniform(float limit) {
    static std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

using ActFn    = std::function<float(float)>;
using ActDeriv = std::function<float(float)>;

Layer::Layer(const LayerConfig& cfg,
          ActFn activation,
          ActDeriv activation_deriv)
        : in_size_(cfg.input_size),
          out_size_(cfg.output_size),
          w_(cfg.weights_ptr),
          b_(cfg.biases_ptr),
          z_(cfg.z_ptr),
          a_(cfg.a_ptr),
          delta_(cfg.delta_ptr),
          grad_w_(cfg.grad_w_ptr),
          grad_b_(cfg.grad_b_ptr),
          act_fn_(std::move(activation)),
          act_drv_(std::move(activation_deriv))
{
    reset_gradients();
}

void Layer::forward(const float* x) {
    for (size_t j = 0; j < out_size_; ++j) {
        float sum = b_[j];
        float* w_row = w_ + j * in_size_;
        for (size_t i = 0; i < in_size_; ++i) {
            sum += w_row[i] * x[i];
        }
        z_[j] = sum;
        a_[j] = act_fn_(sum);
    }
}

void Layer::backward(const float *x, const float *grad_out, float *grad_in) {
    for (size_t j = 0; j < out_size_; ++j) {
        delta_[j] = grad_out[j] * act_drv_(z_[j]);
    }

    for (size_t i = 0; i < in_size_; ++i) {
        float d = delta_[i];
        grad_b_[i] += d;
        float* gw_row = grad_w_ + i * out_size_;
        float* w_row = w_ + i * out_size_;
        for (size_t j = 0; j < out_size_; ++j) {
            gw_row[j] += d * x[i];
            grad_in[i] += w_row[j] * d;
        }
    }
}


