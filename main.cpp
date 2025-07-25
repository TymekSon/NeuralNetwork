#include <iomanip>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include "MINST_Loader.h"
#include "Layer.h"
#include "Arena.h"

int main() {

    //TEST ARENY
    MemoryArena arena(18);

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

    float x[2] = {1.0f,2.0f};

    Layer layer(cfg, ActivationType::Sigmoid);

    layer.forward(x);

    arena.printContent(std::cout);

    // 6. Backward z grad_out = [1, 1]
    float grad_out[2] = {1.0f,2.0f};
    float grad_in[2]  = {0.0f,0.0f};
    layer.backward(x, grad_out, grad_in);

    arena.printContent(std::cout);



    return 0;
}