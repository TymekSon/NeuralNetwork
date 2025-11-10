//
// Created by chomi on 10.11.2025.
//

#include "DenseLayer.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <cstring>

DenseLayer::DenseLayer(size_t in_size, size_t out_size, ActivationType type) {
    in_size_ = in_size;
    out_size_ = out_size;
    act_type_ = type;
}

void DenseLayer::assign_arena_pointers(float*& ptr) {
    w_      = ptr; ptr += in_size_ * out_size_;
    b_      = ptr; ptr += out_size_;
    z_      = ptr; ptr += out_size_;
    a_      = ptr; ptr += out_size_;
    delta_  = ptr; ptr += out_size_;
    grad_w_ = ptr; ptr += in_size_ * out_size_;
    grad_b_ = ptr; ptr += out_size_;
    v_w_    = ptr; ptr += in_size_ * out_size_;
    v_b_    = ptr; ptr += out_size_;

    // Inicjalizacja
    std::fill(b_, b_ + out_size_, 0.0f);
    std::fill(v_w_, v_w_ + in_size_ * out_size_, 0.0f);
    std::fill(v_b_, v_b_ + out_size_, 0.0f);

    // Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (in_size_ + out_size_));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);

    for (size_t i = 0; i < in_size_ * out_size_; ++i) {
        w_[i] = dist(gen);
    }
}

size_t DenseLayer::total_arena_size() const {
    return
        in_size_ * out_size_ * 3 +  // w, grad_w, v_w
        out_size_ * 6;              // b, z, a, delta, grad_b, v_b
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

void DenseLayer::update(float lr, float momentum, size_t batch_size) {
    float inv_batch = 1.0f / (float)batch_size;

    size_t w_count = in_size_ * out_size_;
    for (size_t i = 0; i < w_count; ++i) {
        float grad = grad_w_[i] * inv_batch;
        v_w_[i] = momentum * v_w_[i] - lr * grad;
        w_[i] += v_w_[i];
    }

    for (size_t i = 0; i < out_size_; ++i) {
        float grad = grad_b_[i] * inv_batch;
        v_b_[i] = momentum * v_b_[i] - lr * grad;
        b_[i] += v_b_[i];
    }

    reset_gradients();
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

