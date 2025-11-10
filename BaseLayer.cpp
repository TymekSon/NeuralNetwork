//
// Created by chomi on 10.11.2025.
//

#include "BaseLayer.h"
#include <cmath>

float BaseLayer::activate(float x, ActivationType type) {
    switch(type) {
        case ActivationType::ReLU:     return x > 0.0f ? x : 0.0f;
        case ActivationType::LReLU:    return x > 0.0f ? x : 0.01f * x;
        case ActivationType::Sigmoid:  return 1.0f / (1.0f + std::exp(-x));
        case ActivationType::Tanh:     return std::tanh(x);
        case ActivationType::Identity:
            default:                       return x;
    }
}

float BaseLayer::activate_derivative(float x, ActivationType type) {
    switch(type) {
        case ActivationType::ReLU:     return x > 0.0f ? 1.0f : 0.0f;
        case ActivationType::LReLU:    return x > 0.0f ? 1.0f : 0.01f;
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
        default:                       return 1.0f;
    }
}

