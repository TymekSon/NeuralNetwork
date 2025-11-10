#ifndef BASELAYER_H
#define BASELAYER_H

#include "layer_config.h"
#include <cstddef>

enum class ActivationType { Identity, ReLU, LReLU, Sigmoid, Tanh, Softmax };

class BaseLayer {
public:
  virtual ~BaseLayer() = default;

  virtual void forward(const float* input) = 0;
  virtual void backward(const float* input, const float* grad_out, float* grad_in) = 0;
  virtual void update(float lr, float momentum, size_t batch_size) = 0;
  virtual void reset_gradients() = 0;

  // Nowa metoda: przypisz wskaźniki z areny
  virtual void assign_arena_pointers(float*& ptr) = 0;
  virtual size_t total_arena_size() const = 0;

  virtual void get_input_shape(size_t& h, size_t& w, size_t& c) const { h=w=c=0; }
  virtual void get_output_shape(size_t& h, size_t& w, size_t& c) const { h=w=c=0; }

  // Gettery
  const float* output() const { return a_; }
  const float* pre_activation() const { return z_; }
  size_t input_size() const { return in_size_; }
  size_t output_size() const { return out_size_; }

  // Dane wspólne
  size_t in_size_ = 0;
  size_t out_size_ = 0;
  ActivationType act_type_ = ActivationType::Identity;

  float* w_ = nullptr;      // [in * out]
  float* b_ = nullptr;      // [out]
  float* z_ = nullptr;      // [out]
  float* a_ = nullptr;      // [out]
  float* delta_ = nullptr;  // [out]
  float* grad_w_ = nullptr; // [in * out]
  float* grad_b_ = nullptr; // [out]
  float* v_w_ = nullptr;    // [in * out] – momentum
  float* v_b_ = nullptr;    // [out]

protected:
  static float activate(float x, ActivationType type);
  static float activate_derivative(float x, ActivationType type);
};

#endif // BASELAYER_H