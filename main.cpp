#include <iomanip>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "MINST_Loader.h"
#include "Layer.h"
#include "Arena.h"

int main() {

    //TEST ARENY PAMIECI
    MemoryArena arena(18);

    // 2. Ustaw LayerConfig
    LayerConfig cfg;
    cfg.input_size    = 2;
    cfg.output_size   = 2;
    cfg.weights_ptr   = arena.allocate(2*2); // 4
    cfg.biases_ptr    = arena.allocate(2);   // 2
    cfg.z_ptr         = arena.allocate(2);   // 2
    cfg.a_ptr         = arena.allocate(2);   // 2
    cfg.delta_ptr     = arena.allocate(2);   // 2
    cfg.grad_w_ptr    = arena.allocate(2*2); // 4
    cfg.grad_b_ptr    = arena.allocate(2);   // 2

    float* W = cfg.weights_ptr;
    float* b = cfg.biases_ptr;
    W[0] = 1; W[1] = 2;
    W[2] = 3; W[3] = 4;
    b[0] =  0.5f;
    b[1] = -0.5f;

    float x[2] = {1.0f,1.0f};

    // 4. Stwórz warstwę
    Layer layer(cfg, ActivationType::Sigmoid);

    layer.forward(x);
    const float* out = layer.output_activations();
    // oczekujemy z = [1+2+0.5=3.5, 3+4-0.5=6.5], a = z
    assert(std::fabs(out[0] - 3.5f) < 1e-6);
    assert(std::fabs(out[1] - 6.5f) < 1e-6);

    // 6. Backward z grad_out = [1, 1]
    float grad_out[2] = {1.0f,1.0f};
    float grad_in[2]  = {0.0f,0.0f};
    layer.backward(x, grad_out, grad_in);

    //  grad_w powinno być:
    //   dW[0,0] = 1*1, dW[0,1] = 1*1  → [1,1]
    //   dW[1,0] = 1*1, dW[1,1] = 1*1  → [1,1]
    float* dw = cfg.grad_w_ptr;
    assert(std::fabs(dw[0] - 1.0f) < 1e-6);
    assert(std::fabs(dw[1] - 1.0f) < 1e-6);
    assert(std::fabs(dw[2] - 1.0f) < 1e-6);
    assert(std::fabs(dw[3] - 1.0f) < 1e-6);

    // grad_b = [1,1]
    float* db = cfg.grad_b_ptr;
    assert(std::fabs(db[0] - 1.0f) < 1e-6);
    assert(std::fabs(db[1] - 1.0f) < 1e-6);

    // grad_in: [W[0,0]*1 + W[1,0]*1, W[0,1]*1 + W[1,1]*1] = [1+3, 2+4] = [4,6]
    assert(std::fabs(grad_in[0] - 4.0f) < 1e-6);
    assert(std::fabs(grad_in[1] - 6.0f) < 1e-6);

    std::cout << "All tests passed!\n";
    return 0;
    return 0;
}