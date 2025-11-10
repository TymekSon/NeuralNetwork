//
// Created by chomi on 10.11.2025.
//

#include "DenseLayer.h"
#include <cmath>
#include <algorithm>

static float rand_uniform(float limit = 0.5f) {
    static std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

DenseLayer::DenseLayer(const LayerConfig& cfg, ActivationType type) {
    in_size_ = cfg.input_size;
    out_size_ = cfg.output_size;
    act_type_ = type;

    w_ = cfg.weights_ptr;
    b_ = cfg.biases_ptr;
    z_ = cfg.z_ptr;
    a_ = cfg.a_ptr;
    delta_ = cfg.delta_ptr;
    grad_w_ = cfg.grad_w_ptr;
    grad_b_ = cfg.grad_b_ptr;

    reset_gradients();

    for (size_t i = 0; i < in_size_ * out_size_; ++i)
        w_[i] = rand_uniform();
    std::fill(b_, b_ + out_size_, 0.0f);

    v_w.resize(in_size_ * out_size_, 0.0f);
    v_b.resize(out_size_, 0.0f);
}

void DenseLayer::forward(const float* x) {
    for (size_t j = 0; j < out_size_; ++j) {
        float sum = b_[j];
        const float* w_row = w_ + j * in_size_;
        for (size_t i = 0; i < in_size_; ++i)
            sum += w_row[i] * x[i];
        z_[j] = sum;
    }

    if (act_type_ == ActivationType::Softmax)
        apply_softmax();
    else
        for (size_t j = 0; j < out_size_; ++j)
            a_[j] = activate(z_[j], act_type_);
}

void DenseLayer::backward(const float* x, const float* grad_out, float* grad_in) {
    for (size_t j = 0; j < out_size_; ++j) {
        float d = grad_out[j];
        if (act_type_ != ActivationType::Softmax)
            d *= activate_derivative(z_[j], act_type_);
        delta_[j] = d;
    }

    for (size_t j = 0; j < out_size_; ++j)
        grad_b_[j] += delta_[j];

    for (size_t j = 0; j < out_size_; ++j) {
        float* gw_row = grad_w_ + j * in_size_;
        for (size_t i = 0; i < in_size_; ++i)
            gw_row[i] += delta_[j] * x[i];
    }

    std::fill(grad_in, grad_in + in_size_, 0.0f);
    for (size_t j = 0; j < out_size_; ++j) {
        const float* w_row = w_ + j * in_size_;
        float d = delta_[j];
        for (size_t i = 0; i < in_size_; ++i)
            grad_in[i] += w_row[i] * d;
    }
}

void DenseLayer::reset_gradients() {
    std::fill(grad_w_, grad_w_ + in_size_ * out_size_, 0.0f);
    std::fill(grad_b_, grad_b_ + out_size_, 0.0f);
}

void DenseLayer::apply_softmax() {
    float max_z = z_[0];
    for (size_t j = 1; j < out_size_; ++j)
        if (z_[j] > max_z) max_z = z_[j];

    double sum = 0.0;
    for (size_t j = 0; j < out_size_; ++j) {
        double e = std::exp((double)z_[j] - max_z);
        a_[j] = (float)e;
        sum += e;
    }
    for (size_t j = 0; j < out_size_; ++j)
        a_[j] = (float)((double)a_[j] / sum);
}

