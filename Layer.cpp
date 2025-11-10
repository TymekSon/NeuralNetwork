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
        w_[i] = rand_uniform(0.5f);
    }

    for (int i = 0; i < out_size_; i++) {
        b_[i] = 0.0f;
    }

    v_w.resize(in_size_ * out_size_, 0.0f);
    v_b.resize(out_size_, 0.0f);
}

float Layer::activate(float x, ActivationType type) {
    switch(type) {
        case ActivationType::ReLU:
            return x > 0.0f ? x : 0.0f;
        case ActivationType::LReLU:
            return x > 0.0f ? 1.0f : 0.01f * x;
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
        case ActivationType::LReLU:
            return x > 0.0f ? 1.0f : 0.01f;
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
#ifdef ARENA_DEBUG
    if (!x) throw std::runtime_error("Layer::forward: null input pointer");
    // (opcjonalnie) możesz sprawdzić jakieś granice, ale Layer nie widzi całej areny
#endif
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

void Layer::backward(const float* x, const float* grad_out, float* grad_in) {
    // 1) delta_j = grad_out_j * f'(z_j)  (UWAGA: dla Softmax+CE NIE mnożymy przez f')
    for (size_t j = 0; j < out_size_; ++j) {
        float d = grad_out[j];
        // Jeżeli ta warstwa to softmax użyty z CE i grad_out = (a - y),
        // to delta = grad_out (bez pochodnej). W innym wypadku licz pochodną w z_j:
        if (type_ != ActivationType::Softmax) {
            d *= activate_derivative(z_[j], type_); // pochodna po z, nie po grad_out!
        }
        delta_[j] = d;
    }

    // 2) grad_biasów: sumujemy po j
    for (size_t j = 0; j < out_size_; ++j) {
        grad_b_[j] += delta_[j];
    }

    // 3) grad_wag: dla każdego neuronu wyjściowego j i wejścia i
    //    grad_w[j, i] += delta[j] * x[i]
    for (size_t j = 0; j < out_size_; ++j) {
        float* gw_row = grad_w_ + j * in_size_;   // ten sam layout co w forward
        for (size_t i = 0; i < in_size_; ++i) {
            gw_row[i] += delta_[j] * x[i];
        }
    }

    // 4) grad_in: dL/dx[i] = sum_j w[j,i] * delta[j]
    std::fill(grad_in, grad_in + in_size_, 0.0f);
    for (size_t j = 0; j < out_size_; ++j) {
        const float* w_row = w_ + j * in_size_;
        float d = delta_[j];
        for (size_t i = 0; i < in_size_; ++i) {
            grad_in[i] += w_row[i] * d;
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
