#include "layers.h"

// the Flatten Layer constructor
// input:        -imageHeight(or feature map height)
//               -imageWidth(or feature map width)
//               -imageDepth(or feature map depth)
// output:       N/A
// side effect:  the flatten layer is constructed
// Notes:        N/A
FlattenLayer::FlattenLayer(size_t imageHeight, size_t imageWidth,
                           size_t imageDepth) {
  // make the arrays that carries the flattened data comming from
  // the image or the last convolution layer
  flattened_Arr.assign(imageDepth * imageHeight * imageWidth, 0.0);

  prevLayerGrad.resize(depth);
  for (size_t i = 0; i < depth; i++) {
    prevLayerGrad[i].resize(height);
    for (size_t j = 0; j < height; j++) {
      prevLayerGrad[i][j].assign(width, 0.0);
    }
  }
}

// flattens the incomming image or feature map
// input:        -feature map or image
// output:       N/A
// side effect:  the flattened_Arr is filled with the incomming data
// Note:         if the incomming is an image then it must be 3D
//               or if 2D then the depth is 1
void FlattenLayer::flat(vector<convLayer::featureMapType> &featureMaps) {
  // iterate over each element in the feature maps(or image)
  // by the dimension oreder: depth(feature maps) -> height(rows) ->
  // width(columns)
  for (size_t i = 0; i < featureMaps.size(); i++) {
    for (size_t j = 0; j < featureMaps[0].size(); j++) {
      for (size_t k = 0; k < featureMaps[0][0].size(); k++) {
        // the flatened array is indexed by:
        // i: indexes the feature map
        // j: indexed the rows
        // k: indexed the columns
        flattened_Arr[i * featureMaps[0][0].size() * featureMaps[0].size() +
                      j * featureMaps[0][0].size() + k] = featureMaps[i][j][k];
      }
    }
  }
}

void FlattenLayer::forwardProp(vector<convLayer::featureMapType> &featureMaps){
  flat(featureMaps);
}

void FlattenLayer::backwardProp(vector<double> &nextLayerGrad) {
  // Just reshape the 1D gradient vector back to 3D
  // No math logic (derivative is 1), just data movement.
  
  #pragma omp parallel for collapse(3)
  for (size_t i = 0; i < depth; i++) {
    for (size_t j = 0; j < height; j++) {
      for (size_t k = 0; k < width; k++) {
        // Retrieve from 1D using the same index formula
        size_t index = i * (height * width) + j * width + k;
        prevLayerGrad[i][j][k] = nextLayerGrad[index];
      }
    }
  }
}

void FlattenLayer::backwardProp_batch(vector<double> &nextLayerGrad) {
    backwardProp(nextLayerGrad);    // Same logic as single backwardProp because there are no weights to accumulate
}

void FlattenLayer::applyOptimizer(Optimizer *opt) {
  // Flatten layer has no weights, so do nothing.
}
