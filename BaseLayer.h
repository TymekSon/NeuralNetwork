//
// Created by chomi on 10.11.2025.
//
#include "layer_config.h"

#ifndef BASELAYER_H
#define BASELAYER_H

enum class ActivationType { Identity, ReLU, LReLU, Sigmoid, Tanh, Softmax };

class BaseLayer {
public:
  virtual ~BaseLayer() = default;

  virtual void forward(const float* input) = 0;
  virtual void backward(const float* input, const float* grad_out, float* grad_in) = 0;
  virtual void reset_gradients() = 0;

  const float* output() const { return a_; }
  const float* pre_activation() const { return z_; }

  size_t input_size() const { return in_size_; }
  size_t output_size() const { return out_size_; }

  size_t in_size_{};
  size_t out_size_{};
  ActivationType act_type_{};

  float *z_ = nullptr;
  float *a_ = nullptr;
  float *delta_ = nullptr;

protected:
  static float activate(float x, ActivationType type);
  static float activate_derivative(float x, ActivationType type);
};



#endif //BASELAYER_H
