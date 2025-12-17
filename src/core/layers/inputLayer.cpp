#include "layers.h"

// input Layer constructor
// input:        -imageHeight
//               -imageWidth
//               -imageDepth
// output:       N/A
// side effect:  The input layer is constructed
// Note:         N/A
inputLayer::inputLayer(size_t imageHeight, size_t imageWidth,
                       size_t imageDepth) {
  // intialize the dimensions of the normalized image matrix
  // when an image is fed for training or classifcation
  // it is first normalized and stored in the normalized image matrix
  normalizedImage.resize(imageDepth);
  for (size_t i = 0; i < imageDepth; i++) {
    normalizedImage[i].resize(imageHeight);
    for (size_t j = 0; j < imageHeight; j++) {
      normalizedImage[i][j].assign(imageWidth, 0.0);
    }
  }
}

// start the process of training or classification by taking the input image,
// normalizing it, and storing it in the normalizedImage matrix to be used by
// the next layers
// input:        -inputImage (3D unsigned char vector)
// output:       N/A
// side effect:  the normalizedImage matrix is initialzied by the image after
// normalization Note:         N/A
void inputLayer::start(const image &inputImage) {
  // stor the inputImage in the normlized Image matrix
  std::cout << "DEBUG: inputLayer::start. NormSize: " <<
  normalizedImage.size() << " InputSize: " << inputImage.size() << std::endl;
  for (size_t i = 0; i < normalizedImage.size(); i++) {
    for (size_t j = 0; j < normalizedImage[i].size(); j++) {
      for (size_t k = 0; k < normalizedImage[i][j].size(); k++) {
        // normalize each pixel before storage
        normalizedImage[i][j][k] =
            static_cast<double>(inputImage[i][j][k]) / 255.0;
      }
    }
  }
}
