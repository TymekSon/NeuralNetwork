//
// Created by chomi on 08.07.2025.
//

#include "Layer.h"

static float rand_uniform(float limit) {
    static std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

Layer::Layer(size_t in_dim, size_t out_dim, ActivationType act_type)
    : in_dim(in_dim), out_dim(out_dim), act(act_type),
      W_flat(out_dim * in_dim), b(out_dim),
      dW_flat(out_dim * in_dim), db(out_dim),
      z(out_dim), a(out_dim), delta(out_dim)
{
    float limit = 0.0f;
    if(act == ActivationType::ReLU) {
        limit = std::sqrt(2.0f/static_cast<float>(in_dim));
    } else {
        limit = std::sqrt(2.0f/static_cast<float>(in_dim+out_dim));
    }
    for(size_t i = 0; i < W_flat.size(); i++) {
        W_flat[i] = rand_uniform(limit);
        dW_flat[i] = 0.0f;
    }
    for (size_t j = 0; j < out_dim; ++j) {
        b[j] = 0.0f;
        db[j] = 0.0f;
    }
}


float Layer::activate(float x) const {
    switch (act) {
        case ActivationType::ReLU:    return x > 0.0f ? x : 0.0f;
        case ActivationType::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
        case ActivationType::Tanh:    return std::tanh(x);
        default: throw std::logic_error("Nieznany typ aktywacji");
    }
}

float Layer::activate_derivative(float x) const {
    switch (act) {
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
        default: throw std::logic_error("Nieznany typ aktywacji");
    }
}

std::vector<float> Layer::forward(std::vector<float>& a_prev){
    if(a_prev.size() != in_dim) throw std::runtime_error("Nieprawidłowy wymiar wejścia do warstwy");

    for(size_t i = 0; i < out_dim; i++) {
         float sum = b[i];
         size_t row_off = i*in_dim;
         for(size_t j = 0; j < in_dim; j++) {
             sum += W_flat[row_off + j] * a_prev[j];
         }
         z[i] = sum;
         a[i] = activate(sum);
    }
    return a;
}

std::vector<float> Layer::backward(const std::vector<float>& delta_next, const std::vector<float>& W_next_flat, size_t next_out_dim){
    for(size_t i = 0; i < in_dim; i++) {
        float sum = 0.0f;
        for(size_t j = 0; j < next_out_dim; j++) {
            sum += W_next_flat[j * in_dim + i] * delta_next[j];
        }
        delta[i] = sum * activate_derivative(z[i]);
    }
    return delta;
}

void Layer::compute_gradients(std::vector<float>& a_prev){
    for(size_t i = 0; i < out_dim; i++) {
        db[i] = delta[i];
        size_t row_off = i*in_dim;
        for(size_t j = 0; j < in_dim; j++) {
            dW_flat[row_off + j] = delta[i] * a_prev[j];
        }
    }
}

void Layer::update_params(float learning_rate) {
    for (size_t k = 0; k < W_flat.size(); ++k) {
        W_flat[k] -= learning_rate * dW_flat[k];
    }
    for (size_t j = 0; j < out_dim; ++j) {
        b[j] -= learning_rate * db[j];
    }
}

const std::vector<float>& Layer::get_weights() const {
    return W_flat;
}

const std::vector<float>& Layer::get_biases() const {
    return b;
}

