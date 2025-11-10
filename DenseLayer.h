#ifndef DENSELAYER_H
#define DENSELAYER_H

#include "BaseLayer.h"
#include "layer_config.h"

class DenseLayer : public BaseLayer {
public:
    DenseLayer(size_t in_size, size_t out_size, ActivationType type); // uproszczony konstruktor

    void forward(const float* x) override;
    void backward(const float* x, const float* grad_out, float* grad_in) override;
    void update(float lr, float momentum, size_t batch_size) override;
    void reset_gradients() override;

    void assign_arena_pointers(float*& ptr) override;
    size_t total_arena_size() const override;

private:
    void apply_softmax();
};

#endif // DENSELAYER_H