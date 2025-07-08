#include <iomanip>
#include <iostream>
#include <string>

#include "MINST_Loader.h"
#include "Layer.h"

int main() {

    // TEST LOADERA DANYCH:
    MINST_Loader loader;

    std::string testImagesPath = "../Data/testImages.idx3-ubyte";
    std::string trainImagesPath = "../Data/trainImages.idx3-ubyte";
    std::string testLabelsPath = "../Data/testLabels.idx1-ubyte";
    std::string trainLabelsPath = "../Data/trainLabels.idx1-ubyte";

    std::vector<std::vector<uint8_t>> testImages = loader.load_MINST_Images(testImagesPath);
    std::vector<std::vector<uint8_t>> trainImages = loader.load_MINST_Images(trainImagesPath);

    std::vector<uint8_t> testLabels = loader.load_MINST_Labels(testLabelsPath);
    std::vector<uint8_t> trainLabels = loader.load_MINST_Labels(trainLabelsPath);

    std::vector<std::vector<float>> testImagesParsed = loader.normalize_MINST_Images(testImages);
    std::vector<std::vector<float>> trainImagesParsed = loader.normalize_MINST_Images(trainImages);

    for (int i = 0; i < testImages[12].size(); i++) {
        if (i%28 == 0) std::cout << std::endl;
        std::cout << std::setw(3) << std::setprecision(1) << testImagesParsed[12][i] << " ";
    }


    // TEST WARSTWY:
    Layer testLayer(3, 4, ActivationType::ReLU);
    return 0;
}