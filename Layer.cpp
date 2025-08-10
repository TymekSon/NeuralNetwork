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

Layer::Layer(const LayerConfig& cfg,
          ActivationType type)
        : in_size_(cfg.input_size),
          out_size_(cfg.output_size),
          w_(cfg.weights_ptr),
          b_(cfg.biases_ptr),
          z_(cfg.z_ptr),
          a_(cfg.a_ptr),
          delta_(cfg.delta_ptr),
          grad_w_(cfg.grad_w_ptr),
          grad_b_(cfg.grad_b_ptr),
          type_(type)
{
    reset_gradients();
    for (int i = 0; i < in_size_ * out_size_; i++) {
        w_[i] = rand_uniform(1.0f);
    }

    for (int i = 0; i < out_size_; i++) {
        b_[i] = 0.0f;
    }
}

float Layer::activate(float x, ActivationType type) {
    switch(type) {
        case ActivationType::ReLU:
            return x > 0.0f ? x : 0.0f;
        case ActivationType::Sigmoid: {
            float s = 1.0f / (1.0f + std::exp(-x));
            return s;
        }
        case ActivationType::Tanh:
            return std::tanh(x);
        case ActivationType::Identity:
            default:
                return x;
    }
}

float Layer::activate_derivative(float x, ActivationType type) {
    switch(type) {
        case ActivationType::ReLU:
            return x > 0.0f ? 1.0f : 0.0f;
        case ActivationType::Sigmoid: {
            float s = 1.0f / (1.0f + std::exp(-x));
            return s * (1.0f - s);
        }
        case ActivationType::Tanh: {
            float t = std::tanh(x);
            return 1.0f - t * t;
        }
        case ActivationType::Softmax:
        case ActivationType::Identity:
            default:
                return 1.0f;
    }
}

void Layer::forward(const float* x) {
    for (size_t j = 0; j < out_size_; ++j) {
        float sum = b_[j];
        const float* w_row = w_ + j * in_size_;
        for (size_t i = 0; i < in_size_; ++i) {
            sum += w_row[i] * x[i];
        }
        z_[j] = sum;
    }
    if (type_ == ActivationType::Softmax) {
        apply_softmax();
    } else {
        for (size_t j = 0; j < out_size_; ++j) {
            a_[j] = activate(z_[j], type_);
        }
    }
}

void Layer::backward(const float *x, const float *grad_out, float *grad_in) {
    for (size_t j = 0; j < out_size_; ++j) {
        delta_[j] = grad_out[j] * activate_derivative(grad_out[j], type_);
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

void Layer::reset_gradients() {
    std::fill(grad_w_, grad_w_ + in_size_ * out_size_, 0.0f);
    std::fill(grad_b_, grad_b_ + out_size_,        0.0f);
}

void Layer::apply_softmax() {
    // softmax zapisuje wynik do a_ używając z_ jako wejścia (stosujemy stabilizację)
    float max_z = z_[0];
    for (size_t j = 1; j < out_size_; ++j) if (z_[j] > max_z) max_z = z_[j];

    double sum = 0.0;
    for (size_t j = 0; j < out_size_; ++j) {
        double e = std::exp((double)z_[j] - max_z);
        a_[j] = (float)e;
        sum += e;
    }
    for (size_t j = 0; j < out_size_; ++j) {
        a_[j] = (float)((double)a_[j] / sum);
    }
}
