#pragma once
#include "BaseLayer.h"

class MaxPoolingLayer : public BaseLayer {
public:
    MaxPoolingLayer(size_t pool_h, size_t pool_w, size_t stride = 0);

    void forward(const float* x) override;
    void backward(const float* x, const float* grad_out, float* grad_in) override;
    void update(float, float, size_t) override {}  // brak parametrów
    void reset_gradients() override {}

    size_t input_size() const { return in_h_ * in_w_ * in_c_; }
    size_t output_size() const { return out_h_ * out_w_ * in_c_; }

    void get_input_shape(size_t& h, size_t& w, size_t& c) const override {
        h = in_h_; w = in_w_; c = in_c_;
    }
    void get_output_shape(size_t& h, size_t& w, size_t& c) const override {
        h = out_h_; w = out_w_; c = in_c_;
    }

    void assign_arena_pointers(float*& ptr) override;
    size_t total_arena_size() const override;

private:
    size_t in_c_, in_h_, in_w_;
    size_t ph_, pw_, s_;
    size_t out_h_, out_w_;

    float* a_ = nullptr;
    float* mask_ = nullptr;  // indeksy max
};
