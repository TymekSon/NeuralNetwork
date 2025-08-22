#include "Network.h"
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <iostream>

Network::Network(const std::vector<size_t>& sizes,
                 ActivationType hidden_act,
                 ActivationType output_act)
{
    if (sizes.size() < 2) throw std::runtime_error("Network: need >=2 sizes");
    // policz ile floatów potrzeba (weights, biases, z, a, delta, grad_w, grad_b dla każdej warstwy)
    size_t total = 0;
    max_input_size_ = 0;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        size_t in = sizes[i];
        size_t out = sizes[i+1];
        max_input_size_ = std::max(max_input_size_, in);
        total += in * out; // w
        total += out;      // b
        total += out;      // z
        total += out;      // a
        total += out;      // delta
        total += in * out; // grad_w
        total += out;      // grad_b
    }

    // dodatkowe bufory: input_buffer + 2 tmp grad buffy
    total += sizes[0];          // input_buffer
    total += 2 * max_input_size_; // grad_tmp1, grad_tmp2

    arena_ = MemoryArena(total);

    // alokuj input buffer
    input_buffer_ = arena_.allocate(sizes[0]);
    // wyzeruj input
    std::fill(input_buffer_, input_buffer_ + sizes[0], 0.0f);

    // alokuj tmp grad buffery
    grad_tmp1_ = arena_.allocate(max_input_size_);
    grad_tmp2_ = arena_.allocate(max_input_size_);
    std::fill(grad_tmp1_, grad_tmp1_ + max_input_size_, 0.0f);
    std::fill(grad_tmp2_, grad_tmp2_ + max_input_size_, 0.0f);

    // utwórz warstwy
    layers_.reserve(sizes.size() - 1);
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        LayerConfig cfg;
        cfg.input_size = sizes[i];
        cfg.output_size = sizes[i+1];
        cfg.weights_ptr = arena_.allocate(cfg.input_size * cfg.output_size);
        cfg.biases_ptr  = arena_.allocate(cfg.output_size);
        cfg.z_ptr       = arena_.allocate(cfg.output_size);
        cfg.a_ptr       = arena_.allocate(cfg.output_size);
        cfg.delta_ptr   = arena_.allocate(cfg.output_size);
        cfg.grad_w_ptr  = arena_.allocate(cfg.input_size * cfg.output_size);
        cfg.grad_b_ptr  = arena_.allocate(cfg.output_size);

        // wybór aktywacji: ostatnia warstwa -> output_act
        ActivationType act = (i + 1 == sizes.size() - 1) ? output_act : hidden_act;
        layers_.emplace_back(cfg, act);
    }
}

// forward: input_ptr może być wskaźnikiem do input_buffer_ lub innego bufora,
// ale dla spójności wpisujemy do input_buffer_ i używamy go dalej.
void Network::forward_pass(float* input_ptr) {
    // skopiuj input do arena'owego input_buffer_
    std::memcpy(input_buffer_, input_ptr, sizeof(float) * layers_.front().in_size_);

    const float* x = input_buffer_;
    for (size_t li = 0; li < layers_.size(); ++li) {
        layers_[li].forward(x);
        // wejście do następnej warstwy to a_ bieżącej
        x = layers_[li].a_;
    }
}

void Network::backward_pass(float* label) {
    // grad_out_start -> dla ostatniej warstwy: a - y (softmax + CE)
    Layer& out = layers_.back();
    // użyj grad_tmp1_ jako grad_out początkowo
    float* grad_out = grad_tmp1_;
    float* grad_in  = grad_tmp2_;
    // wyczyść
    std::fill(grad_out, grad_out + out.out_size_, 0.0f);
    std::fill(grad_in, grad_in + max_input_size_, 0.0f);

    for (size_t j = 0; j < out.out_size_; ++j) {
        grad_out[j] = out.a_[j] - label[j]; // dL/da_j dla CE(softmax)
    }

    // iteruj od końca do początku
    for (int li = (int)layers_.size() - 1; li >= 0; --li) {
        Layer& L = layers_[li];
        // wejście x do tej warstwy:
        const float* x = (li == 0) ? input_buffer_ : layers_[li - 1].a_;
        // wyczyść grad_in dla rozmiaru in_size_
        std::fill(grad_in, grad_in + L.in_size_, 0.0f);

        // backward: grad_out (rozmiar out) -> grad_in (rozmiar in)
        L.backward(x, grad_out, grad_in);

        // teraz przygotuj grad_out dla następnej (poprzedniej) warstwy:
        // swap wskaźników: grad_out <- grad_in, grad_in <- grad_out (przy użyciu zamiany pointerów)
        std::swap(grad_out, grad_in);
        // (następna iteracja użyje grad_out zawierającego dL/dx poprzedniej warstwy)
    }
}

void Network::update(float lr, size_t batch_size) {
    for (auto& L : layers_) {
        // wagi
        size_t wcount = L.in_size_ * L.out_size_;
        for (size_t k = 0; k < wcount; ++k) {
            L.w_[k] -= (lr * L.grad_w_[k]) / (float)batch_size;
        }
        // biasy
        for (size_t j = 0; j < L.out_size_; ++j) {
            L.b_[j] -= (lr * L.grad_b_[j]) / (float)batch_size;
        }
        // zeruj gradienty po update
        L.reset_gradients();
    }
}

