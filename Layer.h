#pragma once

#include <cstddef>
#include <functional>
#include "layer_config.h"

class Layer {
public:
    // Typy dla funkcji aktywacji
    using ActFn   = std::function<float(float)>;
    using ActDeriv= std::function<float(float)>;

    // Konstruktor: wskaźniki i rozmiary muszą być wcześniej zaalokowane
    Layer(const LayerConfig& cfg,
          ActFn activation,
          ActDeriv activation_deriv);

    // Forward dla jednego przykładu
    //  x[in_size] → z,a[out_size]
    void forward(const float* x);

    // Backward dla jednego przykładu
    //  grad_out[out_size] → grad_in[in_size], przy okazji zapisuje grad_w, grad_b
    void backward(const float* x, const float* grad_out, float* grad_in);

    // Reset gradientów (przy zbiorczym update)
    void reset_gradients();

    // Gettery
    const float* output_activations() const { return a_; }
    const float* raw_sums()         const { return z_; }

private:
    size_t in_size_;
    size_t out_size_;

    float *w_, *b_;      // wagi, biasy
    float *z_, *a_;      // surowe sumy, aktywacje
    float *delta_;       // δ
    float *grad_w_, *grad_b_; // akumulatory gradientów

    ActFn   act_fn_;
    ActDeriv act_drv_;
};
