//
// Created by chomi on 08.07.2025.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

#ifndef MINST_LOADER_H
#define MINST_LOADER_H



class MINST_Loader {
  private:
    uint32_t read32(std::ifstream& file);

  public:
    MINST_Loader() = default;
    std::vector<std::vector<uint8_t>> load_MINST_Images(std::string& file);
    std::vector<uint8_t> load_MINST_Labels(std::string& file);
    std::vector<std::vector<float>> normalize_MINST_Images(const std::vector<std::vector<uint8_t>>& raw_images);
    std::vector<std::vector<float>> to_one_hot(const std::vector<uint8_t>& labels, int num_classes = 10);
};



#endif //MINST_LOADER_H
