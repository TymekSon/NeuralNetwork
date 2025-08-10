//
// Created by chomi on 10.08.2025.
//
#include <vector>
#include <cstddef>
#include <string>
#include "Layer.h"
#include "Arena.h"

#ifndef NETWORK_H
#define NETWORK_H



class Network {
public:
  Network(const std::vector<size_t>& sizes,
          ActivationType hidden_act = ActivationType::ReLU,
          ActivationType output_act = ActivationType::Softmax) {};

  void forward_pass(float* input);

  void backward_pass(float* label);

  void update(float learning_rate, size_t batch_size = 1);

    float* output_ptr() { return layers_.back().a_; }
    size_t output_size() const { return layers_.back().out_size_; }

private:
    std::vector<Layer> layers_;
    MemoryArena arena_ = MemoryArena(1000);
    float* input_buffer_; // wsk na buffer wejściowy (w arenie)
    float* grad_tmp1_;    // bufory pomocnicze (alokowane w arenie)
    float* grad_tmp2_;
    size_t max_input_size_;
};



#endif //NETWORK_H
