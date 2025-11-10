// MaxPoolingLayer.cpp
#include "MaxPoolingLayer.h"
#include <algorithm>
#include <cstring>
#include <limits>

MaxPoolingLayer::MaxPoolingLayer(size_t ph, size_t pw, size_t s)
    : ph_(ph), pw_(pw), s_(s == 0 ? ph : s)
{
}

size_t MaxPoolingLayer::total_arena_size() const {
    size_t out_vol = out_h_ * out_w_ * in_c_;
    return out_vol * 2;  // a_ + mask_ (oba float)
}

void MaxPoolingLayer::assign_arena_pointers(float*& ptr) {
    size_t out_vol = out_h_ * out_w_ * in_c_;
    a_ = ptr; ptr += out_vol;
    mask_ = ptr; ptr += out_vol;

    std::fill(a_, a_ + out_vol, 0.0f);
    std::fill(mask_, mask_ + out_vol, -1.0f);  // -1.0f jako brak
}

void MaxPoolingLayer::forward(const float* x) {
    const float NEG_INF = std::numeric_limits<float>::lowest();

    for (size_t c = 0; c < in_c_; ++c) {
        for (size_t oh = 0; oh < out_h_; ++oh) {
            for (size_t ow = 0; ow < out_w_; ++ow) {
                size_t out_idx = c * out_h_ * out_w_ + oh * out_w_ + ow;
                float max_val = NEG_INF;
                int max_idx = -1;
                for (size_t ky = 0; ky < ph_; ++ky) {
                    for (size_t kx = 0; kx < pw_; ++kx) {
                        int iy = (int)(oh * s_) + ky;
                        int ix = (int)(ow * s_) + kx;
                        if (iy < (int)in_h_ && ix < (int)in_w_) {
                            size_t idx = c * in_h_ * in_w_ + iy * in_w_ + ix;
                            if (x[idx] > max_val) {
                                max_val = x[idx];
                                max_idx = static_cast<float>(idx);
                            }
                        }
                    }
                }
                a_[out_idx] = max_val;
                mask_[out_idx] = max_idx;
            }
        }
    }
}

void MaxPoolingLayer::backward(const float* x, const float* grad_out, float* grad_in) {
    if (grad_in) {
        std::fill(grad_in, grad_in + in_h_ * in_w_ * in_c_, 0.0f);
        for (size_t i = 0; i < out_h_ * out_w_ * in_c_; ++i) {
            int idx = static_cast<int>(mask_[i]);
            if (idx >= 0) {
                grad_in[idx] += grad_out[i];
            }
        }
    }
}