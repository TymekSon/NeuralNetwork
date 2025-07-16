#include "Arena.h"
#include <cstring>

MemoryArena::MemoryArena(size_t total_floats)
    : capacity_(total_floats), offset_(0), used_(0), peak_(0)
{
    data_ = new float[capacity_];
    std::memset(data_, 0, capacity_ * sizeof(float));
}

MemoryArena::~MemoryArena() {
    delete[] data_;
}

float* MemoryArena::allocate(size_t n) {
    if (offset_ + n > capacity_) {
        throw std::runtime_error("MemoryArena: allocation overflow");
    }

    float* ptr = data_ + offset_;
    offset_ += n;
    used_ += n;
    peak_ = std::max(peak_, used_);
    return ptr;
}

void MemoryArena::reset() {
    offset_ = 0;
    used_ = 0;
}

ArenaStats MemoryArena::stats() const {
    return { capacity_, used_, peak_ };
}
