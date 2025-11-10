#include "Network.h"
#include "Arena.h"
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <iostream>

size_t max_width_of(const std::vector<size_t>& sizes) {
        size_t m = 0;
        for (size_t i = 0; i + 1 < sizes.size(); ++i) {
            m = std::max({m, sizes[i], sizes[i+1]});
        }
        return m;
    }

size_t calc_total_for_arena(const std::vector<size_t>& sizes) {
    size_t total = 0;
    size_t maxw  = 0;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        const size_t in  = sizes[i];
        const size_t out = sizes[i+1];
        maxw = std::max({maxw, in, out});

        total += in * out; // w
        total += out;      // b
        total += out;      // z
        total += out;      // a
        total += out;      // delta
        total += in * out; // grad_w
        total += out;      // grad_b
        total += in * out; // v_w   ← DODAJ
        total += out;      // v_b   ← DODAJ
    }
    total += sizes[0];      // input_buffer
    total += 2 * maxw;      // grad_tmp1, grad_tmp2
    return total;
}

Network::Network(std::vector<std::unique_ptr<BaseLayer>> layers, size_t max_input_size)
    : layers_(std::move(layers)),
      max_input_size_(max_input_size),
      input_buffer_(nullptr),
      grad_tmp1_(nullptr),
      grad_tmp2_(nullptr)
{
    if (layers_.empty()) throw std::runtime_error("Network: no layers");

    // === 1. Zbierz rozmiary ===
    std::vector<size_t> sizes;
    sizes.push_back(layers_[0]->input_size());
    for (const auto& l : layers_) {
        sizes.push_back(l->output_size());
    }

    // === 2. max_input_size_ ===
    if (max_input_size_ == 0) {
        max_input_size_ = max_width_of(sizes);
    }

    // === 3. Arena ===
    arena_ = std::make_unique<MemoryArena>(calc_total_for_arena(sizes));

    // === 4. Alokuj bufory ===
    input_buffer_ = arena_->allocate(sizes[0]);
    grad_tmp1_    = arena_->allocate(max_input_size_);
    grad_tmp2_    = arena_->allocate(max_input_size_);

    std::fill(input_buffer_, input_buffer_ + sizes[0], 0.0f);
    std::fill(grad_tmp1_, grad_tmp1_ + max_input_size_, 0.0f);
    std::fill(grad_tmp2_, grad_tmp2_ + max_input_size_, 0.0f);

    // === 5. Przypisz warstwy ===
    float* ptr = arena_->get_current_ptr();
    for (auto& layer : layers_) {
        layer->assign_arena_pointers(ptr);  // ptr się przesuwa!
    }

    arena_->validate_allocations();
}

// forward: input_ptr może być wskaźnikiem do input_buffer_ lub innego bufora,
// ale dla spójności wpisujemy do input_buffer_ i używamy go dalej.
void Network::forward_pass(float* input_ptr) {
    // skopiuj input do arena'owego input_buffer_
    std::memcpy(input_buffer_, input_ptr, sizeof(float) * layers_.front()->in_size_);

    const float* x = input_buffer_;
    for (size_t li = 0; li < layers_.size(); ++li) {
        layers_[li]->forward(x);
        // wejście do następnej warstwy to a_ bieżącej
        x = layers_[li]->a_;
    }
}

void Network::backward_pass(float* label) {
    // grad_out_start -> dla ostatniej warstwy: a - y (softmax + CE)
    auto& out = layers_.back();
    // użyj grad_tmp1_ jako grad_out początkowo
    float* grad_out = grad_tmp1_;
    float* grad_in  = grad_tmp2_;
    // wyczyść
    std::fill(grad_out, grad_out + out->out_size_, 0.0f);
    std::fill(grad_in, grad_in + max_input_size_, 0.0f);

    for (size_t j = 0; j < out->out_size_; ++j) {
        grad_out[j] = out->a_[j] - label[j]; // dL/da_j dla CE(softmax)
    }

    // iteruj od końca do początku
    for (int li = (int)layers_.size() - 1; li >= 0; --li) {
        auto& L = layers_[li];

        const float* x = (li == 0) ? input_buffer_ : layers_[li - 1]->a_;

        std::fill(grad_in, grad_in + L->in_size_, 0.0f);

        L->backward(x, grad_out, grad_in);

        std::swap(grad_out, grad_in);
    }
}

// Network.cpp
void Network::update(float lr, float momentum, size_t batch_size) {
    for (auto& layer : layers_) {
        layer->update(lr, momentum, batch_size);  // <- tylko tyle!
    }
}

void Network::print_stats() {
  arena_->stats();
}
