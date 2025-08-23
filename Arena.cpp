#include "Arena.h"
#include <cstring>

#ifdef ARENA_DEBUG
    // rejestrujemy pary (start_index, length) w jednostkach floatów
    std::vector<std::pair<size_t,size_t>> allocations_;
#endif

MemoryArena::MemoryArena(size_t total_floats)
    : capacity_(total_floats), offset_(0), used_(0), peak_(0)
{
    if (capacity_ == 0) {
        data_ = nullptr;
    } else {
        data_ = new float[capacity_];
        // w debugzie wypełniamy NaNami by łatwiej znaleźć nadpisania
#ifdef ARENA_DEBUG
        for (size_t i = 0; i < capacity_; ++i) data_[i] = std::numeric_limits<float>::quiet_NaN();
#else
        std::memset(data_, 0, capacity_ * sizeof(float));
#endif
    }
}

MemoryArena::~MemoryArena() {
    delete[] data_;
}

float* MemoryArena::allocate(size_t n) {
    if (offset_ + n > capacity_) {
        throw std::runtime_error("MemoryArena: allocation overflow");
    }

    float* ptr = data_ + offset_;

#ifdef ARENA_DEBUG
    // zapamiętaj zakres (start offset, length)
    allocations_.emplace_back(offset_, n);
    // opcjonalnie w debugu wyczyść alokowany region (ustaw 0)
    for (size_t i = 0; i < n; ++i) ptr[i] = 0.0f;
#endif

    offset_ += n;
    used_ += n;
    peak_ = std::max(peak_, used_);
    return ptr;
}

MemoryArena::MemoryArena(MemoryArena&& other) noexcept
    : data_(other.data_), capacity_(other.capacity_),
      offset_(other.offset_), used_(other.used_), peak_(other.peak_)
{
#ifdef ARENA_DEBUG
    allocations_ = std::move(other.allocations_);
#endif
    other.data_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
    other.used_ = 0;
    other.peak_ = 0;
}

// Move assignment
MemoryArena& MemoryArena::operator=(MemoryArena&& other) noexcept {
    if (this != &other) {
        delete[] data_; // zwalniamy stare dane

#ifdef ARENA_DEBUG
        allocations_.clear();
        allocations_ = std::move(other.allocations_);
#endif

        data_ = other.data_;
        capacity_ = other.capacity_;
        offset_ = other.offset_;
        used_ = other.used_;
        peak_ = other.peak_;

        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
        other.used_ = 0;
        other.peak_ = 0;
    }
    return *this;
}

void MemoryArena::reset() {
    offset_ = 0;
    used_ = 0;
#ifdef ARENA_DEBUG
    allocations_.clear();
    // w debugu warto przywrócić NaNy
    if (data_) {
        for (size_t i = 0; i < capacity_; ++i) data_[i] = std::numeric_limits<float>::quiet_NaN();
    }
#endif
}

ArenaStats MemoryArena::stats() const {
    return { capacity_, used_, peak_ };
}

void MemoryArena::printContent(std::ostream& os){
    os << "MemoryArena Content (used=" << used_ << ", capacity=" << capacity_ << "):\n";
    for (size_t i = 0; i < used_; ++i) {
        os << data_[i];
        if (i + 1 < used_) os << ", ";
    }
    os << std::endl;
}

void MemoryArena::validate_allocations() const {
#ifdef ARENA_DEBUG
    // sprawdź nakładanie się zakresów
    for (size_t i = 0; i < allocations_.size(); ++i) {
        size_t s1 = allocations_[i].first;
        size_t e1 = s1 + allocations_[i].second; // exclusive
        for (size_t j = i + 1; j < allocations_.size(); ++j) {
            size_t s2 = allocations_[j].first;
            size_t e2 = s2 + allocations_[j].second;
            if (!(e1 <= s2 || e2 <= s1)) {
                std::cerr << "MemoryArena: OVERLAP detected between alloc " << i << " (" << s1 << "," << e1 << ") and "
                          << j << " (" << s2 << "," << e2 << ")\n";
                throw std::runtime_error("MemoryArena: allocation overlap detected (debug)");
            }
        }
    }

    // dodatkowo sprawdzamy, czy ktokolwiek nadpisał sentinel NaNy poza zarejestrowanymi alokacjami
    // zrobimy to prostym sposobem: stwórz wektor bool rozmiaru capacity, oznacz zajęte miejsca
    std::vector<char> used(capacity_, 0);
    for (auto &p : allocations_) {
        size_t s = p.first;
        size_t len = p.second;
        for (size_t k = 0; k < len; ++k) {
            used[s + k] = 1;
        }
    }
    // sprawdź czy istnieje indeks < used_ który nie jest oznaczony (to podejrzane)
    for (size_t idx = 0; idx < capacity_; ++idx) {
        if (used[idx]) continue;
        float v = data_[idx];
        if (!std::isnan(v) && idx < used_) {
            std::cerr << "MemoryArena: WARNING non-nan value at idx " << idx << " outside allocations (value=" << v << ")\n";
            // nie rzucamy od razu, ale informujemy; można też throw
        }
    }
#endif
}
