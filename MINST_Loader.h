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
    MINST_Loader();
    ~MINST_Loader();
    std::vector<std::vector<uint8_t>> load_MINST_Images(std::string& file);
    std::vector<uint8_t> load_MINST_Labels(std::string& file);
    std::vector<std::vector<bool>> parse_MINST_Images(std::vector<std::vector<uint8_t>>& images, float threshold, std::string& file);
};



#endif //MINST_LOADER_H
