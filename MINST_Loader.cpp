//
// Created by chomi on 08.07.2025.
//

#include "MINST_Loader.h"

// Loading Images for training and testing
std::vector<std::vector<uint8_t>> MINST_Loader::load_MINST_Images(std::string& file){
    std::ifstream input_file(file, std::ios::binary);
    if (!input_file) throw std::runtime_error("Nie można otworzyć pliku obrazów!");

    uint32_t magic = read32(input_file);
    uint32_t count = read32(input_file);
    uint32_t rows = read32(input_file);
    uint32_t cols = read32(input_file);

    // creating vector of vectors, each containing one image in a flat array
    std::vector<std::vector<uint8_t>> images(count, std::vector<uint8_t>(rows * cols));
    for(uint32_t i = 0; i < count; i++){
        input_file.read((char*)images[i].data(), rows * cols);
    }
    return images;
}

// Loading Labels to compare network output
std::vector<uint8_t> MINST_Loader::load_MINST_Labels(std::string& file){
    std::ifstream input_file(file, std::ios::binary);
    if(!input_file) throw std::runtime_error("Nie można otworzyć pliku etykiet!");

    uint32_t magic = read32(input_file);
    uint32_t count = read32(input_file);

    std::vector<uint8_t> labels(count);
    input_file.read((char*)labels.data(), count);
    return labels;
}

// Parsing pixel values from 0-255 to 0-1
std::vector<std::vector<bool>> MINST_Loader::parse_MINST_Images(std::vector<std::vector<uint8_t>>& images, float threshold, std::string& file){
    std::ifstream input_file(file, std::ios::binary);

    uint32_t magic = read32(input_file);
    uint32_t count = read32(input_file);
    uint32_t rows = read32(input_file);
    uint32_t cols = read32(input_file);

    std::vector<std::vector<bool>> images_Parsed(count, std::vector<bool>(rows * cols));
    for(uint32_t i = 0; i < images.size(); i++){
        for(uint32_t j = 0; j < images[i].size(); j++){
            if(images[i][j] >= threshold){
                images_Parsed[i][j] = 1;
            }
            else{
                images_Parsed[i][j] = 0;
            }
        }
    }
    return images_Parsed;
}

// Loading image data(size, datatype) for other methods
uint32_t MINST_Loader::read32(std::ifstream& file){
    uint8_t bytes[4];
    file.read((char*)&bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}
