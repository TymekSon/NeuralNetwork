//
// Created by chomi on 10.11.2025.
//
#include "BaseLayer.h"
#include "layer_config.h"
#include <vector>
#include <algorithm>
#include <random>

#ifndef DENSELAYER_H
#define DENSELAYER_H



class DenseLayer : public BaseLayer {
public:
    DenseLayer(const LayerConfig& cfg, ActivationType type);

    void forward(const float* x) override;
    void backward(const float* x, const float* grad_out, float* grad_in) override;
    void reset_gradients() override;

private:
    void apply_softmax();

    float* w_ = nullptr;
    float* b_ = nullptr;
    float* grad_w_ = nullptr;
    float* grad_b_ = nullptr;

    std::vector<float> v_w;
    std::vector<float> v_b;
};



#endif //DENSELAYER_H
