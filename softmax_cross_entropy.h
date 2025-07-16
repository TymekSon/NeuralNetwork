//
// Created by chomi on 16.07.2025.
//

#ifndef SOFTMAX_CROSS_ENTROPY_H
#define SOFTMAX_CROSS_ENTROPY_H

#include <cstddef>

class softmax_cross_entropy {
public:
    // Forward: z[len] – logity, y[len] – one-hot soft labels (0 lub 1)
    // zwraca loss
    static float forward(const float* z, const float* y, float* p, size_t len);

    // Backward: p[len] – probabilties z poprzedniego forward
    // y[len] – one-hot; zwraca gradient w grad_z[len]
    static void backward(const float* p, const float* y, float* grad_z, size_t len);
};



#endif //SOFTMAX_CROSS_ENTROPY_H
