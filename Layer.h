//
// Created by chomi on 08.07.2025.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cstddef>
#include <random>
#include <cmath>
#include <stdexcept>

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh
};

class Layer {
  public:
    Layer(size_t in_dim, size_t out_dim, ActivationType act);

    std::vector<float> forward(std::vector<float>& a_prev);

    std::vector<float> backward(const std::vector<float>& delta_next, const std::vector<float>& a_next, size_t next_out_dim);

    void compute_gradients(std::vector<float>& a_prev);

    void update_params(float learning_rate);

    float activate(float x) const;

    float activate_derivative(float x) const;

    const std::vector<float>& get_weights() const;
    const std::vector<float>& get_biases() const;

private:
    size_t in_dim, out_dim;
    std::vector<float> W_flat, b, dW_flat, db, z, a, delta;
    ActivationType act;
};



#endif //LAYER_H
