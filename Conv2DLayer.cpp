#include "Conv2DLayer.h"
#include <random>
#include <cmath>
#include <cstring>
#include <stdexcept>

Conv2DLayer::Conv2DLayer(
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_c, size_t k, size_t s, size_t p,
    ActivationType act
) : in_c_(in_c), in_h_(in_h), in_w_(in_w),
    out_c_(out_c), k_(k), s_(s), p_(p), act_(act)
{
    if (k_ == 0 || s_ == 0) throw std::invalid_argument("kernel/stride > 0");
    out_h_ = (in_h_ + 2*p_ - k_) / s_ + 1;
    out_w_ = (in_w_ + 2*p_ - k_) / s_ + 1;
    if (out_h_ <= 0 || out_w_ <= 0) throw std::invalid_argument("invalid output size");
}

size_t Conv2DLayer::total_arena_size() const {
    size_t k_vol = out_c_ * in_c_ * k_ * k_;
    size_t out_vol = out_h_ * out_w_ * out_c_;
    return k_vol * 3 + out_c_ * 3 + out_vol * 3;  // kernel, grad_k, v_k, bias, grad_b, v_b, z, a, delta
}

void Conv2DLayer::assign_arena_pointers(float*& ptr) {
    size_t k_vol = out_c_ * in_c_ * k_ * k_;
    size_t out_vol = out_h_ * out_w_ * out_c_;

    kernel_  = ptr; ptr += k_vol;
    bias_    = ptr; ptr += out_c_;
    z_       = ptr; ptr += out_vol;
    a_       = ptr; ptr += out_vol;
    delta_   = ptr; ptr += out_vol;
    grad_k_  = ptr; ptr += k_vol;
    grad_b_  = ptr; ptr += out_c_;
    v_k_     = ptr; ptr += k_vol;
    v_b_     = ptr; ptr += out_c_;

    std::fill(bias_, bias_ + out_c_, 0.0f);
    std::fill(v_k_, v_k_ + k_vol, 0.0f);
    std::fill(v_b_, v_b_ + out_c_, 0.0f);

    // Xavier init
    float scale = std::sqrt(2.0f / (in_c_ * k_ * k_));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < k_vol; ++i) kernel_[i] = dist(gen);
}

void Conv2DLayer::im2col(const float* x, float* cols) const {
    const size_t cols_per_patch = in_c_ * k_ * k_;
    const size_t patches = out_h_ * out_w_;

    for (size_t oh = 0; oh < out_h_; ++oh) {
        for (size_t ow = 0; ow < out_w_; ++ow) {
            size_t col_idx = (oh * out_w_ + ow) * cols_per_patch;
            for (size_t c = 0; c < in_c_; ++c) {
                for (size_t ky = 0; ky < k_; ++ky) {
                    for (size_t kx = 0; kx < k_; ++kx) {
                        int iy = (int)(oh * s_) + ky - p_;
                        int ix = (int)(ow * s_) + kx - p_;
                        if (iy >= 0 && iy < (int)in_h_ && ix >= 0 && ix < (int)in_w_) {
                            cols[col_idx++] = x[c * in_h_ * in_w_ + iy * in_w_ + ix];
                        } else {
                            cols[col_idx++] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void Conv2DLayer::forward(const float* x) {
    const size_t patches = out_h_ * out_w_;
    const size_t k_vol = in_c_ * k_ * k_;
    const size_t cols_size = patches * k_vol;

    // Przygotuj im2col (tymczasowy bufor w arenie – można zoptymalizować)
    std::vector<float> cols(cols_size);
    im2col(x, cols.data());

    // z = W @ cols + b
    for (size_t oc = 0; oc < out_c_; ++oc) {
        for (size_t p = 0; p < patches; ++p) {
            float sum = bias_[oc];
            const float* w = kernel_ + oc * k_vol;
            const float* col = cols.data() + p * k_vol;
            for (size_t i = 0; i < k_vol; ++i) {
                sum += w[i] * col[i];
            }
            z_[oc * patches + p] = sum;
        }
    }

    // a = act(z)
    for (size_t i = 0; i < out_h_ * out_w_ * out_c_; ++i) {
        a_[i] = BaseLayer::activate(z_[i], act_);
    }
}

void Conv2DLayer::backward(const float* x, const float* grad_out, float* grad_in) {
    const size_t patches = out_h_ * out_w_;
    const size_t k_vol = in_c_ * k_ * k_;

    std::vector<float> cols(k_vol * patches);
    im2col(x, cols.data());

    // delta = grad_out * act'(z)
    for (size_t i = 0; i < out_h_ * out_w_ * out_c_; ++i) {
        float d = grad_out[i];
        if (act_ != ActivationType::Softmax) {
            d *= BaseLayer::activate_derivative(z_[i], act_);
        }
        delta_[i] = d;
    }

    // grad_b
    for (size_t oc = 0; oc < out_c_; ++oc) {
        float sum = 0.0f;
        for (size_t p = 0; p < patches; ++p) {
            sum += delta_[oc * patches + p];
        }
        grad_b_[oc] += sum;
    }

    // grad_k = delta @ cols^T
    for (size_t oc = 0; oc < out_c_; ++oc) {
        float* gk = grad_k_ + oc * k_vol;
        for (size_t p = 0; p < patches; ++p) {
            float d = delta_[oc * patches + p];
            const float* col = cols.data() + p * k_vol;
            for (size_t i = 0; i < k_vol; ++i) {
                gk[i] += d * col[i];
            }
        }
    }

    // grad_in = W^T @ delta (col2im)
    if (grad_in) {
        std::fill(grad_in, grad_in + in_h_ * in_w_ * in_c_, 0.0f);
        for (size_t oc = 0; oc < out_c_; ++oc) {
            const float* w = kernel_ + oc * k_vol;
            for (size_t p = 0; p < patches; ++p) {
                float d = delta_[oc * patches + p];
                size_t oh = p / out_w_, ow = p % out_w_;
                for (size_t c = 0; c < in_c_; ++c) {
                    for (size_t ky = 0; ky < k_; ++ky) {
                        for (size_t kx = 0; kx < k_; ++kx) {
                            int iy = (int)(oh * s_) + ky - p_;
                            int ix = (int)(ow * s_) + kx - p_;
                            if (iy >= 0 && iy < (int)in_h_ && ix >= 0 && ix < (int)in_w_) {
                                size_t idx = c * in_h_ * in_w_ + iy * in_w_ + ix;
                                size_t kidx = c * k_ * k_ + ky * k_ + kx;
                                grad_in[idx] += w[kidx] * d;
                            }
                        }
                    }
                }
            }
        }
    }
}

void Conv2DLayer::update(float lr, float momentum, size_t batch_size) {
    float inv = 1.0f / batch_size;
    size_t k_vol = out_c_ * in_c_ * k_ * k_;

    for (size_t i = 0; i < k_vol; ++i) {
        float g = grad_k_[i] * inv;
        v_k_[i] = momentum * v_k_[i] - lr * g;
        kernel_[i] += v_k_[i];
    }
    for (size_t i = 0; i < out_c_; ++i) {
        float g = grad_b_[i] * inv;
        v_b_[i] = momentum * v_b_[i] - lr * g;
        bias_[i] += v_b_[i];
    }
    reset_gradients();
}

void Conv2DLayer::reset_gradients() {
    size_t k_vol = out_c_ * in_c_ * k_ * k_;
    std::fill(grad_k_, grad_k_ + k_vol, 0.0f);
    std::fill(grad_b_, grad_b_ + out_c_, 0.0f);
}