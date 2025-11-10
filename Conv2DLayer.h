//
// Created by chomi on 10.11.2025.
//
#include"BaseLayer.h"

#ifndef CONV2DLAYER_H
#define CONV2DLAYER_H



class Conv2DLayer : public BaseLayer {
public:
    Conv2DLayer(
        size_t in_channels,
        size_t in_height,
        size_t in_width,
        size_t out_channels,
        size_t kernel_size,
        size_t stride = 1,
        size_t padding = 0,
        ActivationType act = ActivationType::ReLU
    );

    void forward(const float* x) override;
    void backward(const float* x, const float* grad_out, float* grad_in) override;
    void update(float lr, float momentum, size_t batch_size) override;
    void reset_gradients() override;

    size_t input_size() const { return in_h_ * in_w_ * in_c_; }
    size_t output_size() const { return out_h_ * out_w_ * out_c_; }

    void get_input_shape(size_t& h, size_t& w, size_t& c) const override {
        h = in_h_; w = in_w_; c = in_c_;
    }
    void get_output_shape(size_t& h, size_t& w, size_t& c) const override {
        h = out_h_; w = out_w_; c = out_c_;
    }

    void assign_arena_pointers(float*& ptr) override;
    size_t total_arena_size() const override;

private:
    size_t in_c_, in_h_, in_w_;
    size_t out_c_, out_h_, out_w_;
    size_t k_, s_, p_;
    ActivationType act_;

    // Bufory w arenie
    float* kernel_ = nullptr;      // [out_c][in_c][k][k]
    float* bias_ = nullptr;        // [out_c]
    float* z_ = nullptr;           // [out_h][out_w][out_c]
    float* a_ = nullptr;           // [out_h][out_w][out_c]
    float* delta_ = nullptr;       // [out_h][out_w][out_c]
    float* grad_k_ = nullptr;      // [out_c][in_c][k][k]
    float* grad_b_ = nullptr;      // [out_c]
    float* v_k_ = nullptr;         // momentum
    float* v_b_ = nullptr;

    void im2col(const float* x, float* cols) const;
    void col2im(const float* cols, float* dx) const;
};



#endif //CONV2DLAYER_H
