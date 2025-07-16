#pragma once
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <iostream>

struct ArenaStats {
    size_t capacity;
    size_t used;
    size_t peak;
};

class MemoryArena {
public:
    explicit MemoryArena(size_t total_floats);
    ~MemoryArena();

    float* allocate(size_t n);      // Zwraca wskaźnik do n floatów
    void reset();                   // Zaczyna alokację od początku
    ArenaStats stats() const;

private:
    float* data_;
    size_t capacity_;  // liczba floatów
    size_t offset_;    // aktualna pozycja alokacji
    size_t used_;
    size_t peak_;
};
