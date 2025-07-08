//
// Created by chomi on 08.07.2025.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cstddef>


enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh
};

class Layer {
  public:
    Layer(size_t layerSize, size_t inputSize, ActivationType type = ActivationType::ReLU);

    std::vector<float> forward(std::vector<float>& a_prev);

    std::vector<float> backward(std::vector<float>& delta_next, std::vector<float>& a_next);

    void compute_gradients(std::vector<float>& a_prev, std::vector<float>& a_next);

    void update_params(float learning_rate);

    const std::vector<float>& get_weights() const;
    const std::vector<float>& get_biases() const;

  private:
    size_t layerSize, inputSize;
    ActivationType activationType;

    std::vector<float> weights_flat;
    std::vector<float> biases;
    std::vector<float> weight_gradients_flat;
    std::vector<float> bias_gradients;

    std::vector<float> z_;
    std::vector<float> a_;
    std::vector<float> delta_;

    float activate(float x) const;
    float activate_derivative(float x) const;
};



#endif //LAYER_H
