#include "layers.h"

// pooling layer constructor
// input:                -kernelConfig (dimensions of the filter and number of
// strides)
//                       -FM_Dim (dimensions of the output feature map)
//                       -poolType (max or average)
// output:               N/A
// side effect:          the pooling layer is constructed
// Note:                 N/A
poolingLayer::poolingLayer(poolKernel &kernelConfig, featureMapDim &FM_Dim,
                           poolingLayerType &poolType)
    : fm(FM_Dim), kernel_info(kernelConfig), poolingType(poolType) {
  // initialize the feature map to all zeros
  featureMaps.resize(FM_Dim.FM_depth);
  for (size_t i = 0; i < FM_Dim.FM_depth; i++) {
    featureMaps[i].resize(FM_Dim.FM_height);
    for (size_t j = 0; j < FM_Dim.FM_height; j++) {
      featureMaps[i][j].assign(FM_Dim.FM_width, 0.0);
    }
  }

  size_t inputHeight =
      kernelConfig.filter_height + kernelConfig.stride * (fm.FM_height - 1);
  size_t inputWidth =
      kernelConfig.filter_width + kernelConfig.stride * (fm.FM_width - 1);
  size_t inputDepth = kernelConfig.filter_depth;

  prevLayerGrad.resize(inputDepth);
  for (size_t i = 0; i < inputDepth; i++) {
    prevLayerGrad[i].resize(inputHeight);
    for (size_t j = 0; j < inputHeight; j++) {
      prevLayerGrad[i][j].assign(inputWidth, 0.0);
    }
  }
}

// forward propagation of the pooling layer
// done by applying max or average pooling to the feature maps
// input:                inputFeatureMaps
// output:               N/A
// side effect:          the output feature map is filled with the result of the
// pooling Note:                 N/A
void poolingLayer::forwardProp(vector<featureMapType> &inputFeatureMaps) {
  size_t out_FM_height_Iter = 0; // iterates the rows of the output feature map
  size_t out_FM_width_Iter = 0;  // iterates the columns of the output feature
                                 // map

  // iterate on every channel (depth) using 'd', where 'd' is the depth
  // corrisponding of the kernel depth and the input feature map depth, and
  // output feature map depth
  for (size_t d = 0; d < kernel_info.filter_depth; d++) {
    if (d >= inputFeatureMaps.size())
      continue;

    // moving on the 2D feature map the length to be moved is fm_dim - krnl_dim
    // + 1
    size_t inputH = inputFeatureMaps[d].size();

    // Safety check: if input is smaller than filter, skip or handle gracefully
    if (inputH < kernel_info.filter_height) {
      continue;
    }

    size_t h_limit = inputH - kernel_info.filter_height + 1;

    // move along the rows of the input feature map using 'i', and jump with
    // amount "stride"
    for (size_t i = 0; i < h_limit; i += kernel_info.stride) {
      if (i >= inputFeatureMaps[d].size())
        break; // Safety

      // Safety check for width
      size_t inputW = inputFeatureMaps[d][i].size();
      if (inputW < kernel_info.filter_width) {
        continue;
      }

      size_t w_limit = inputW - kernel_info.filter_width + 1;

      // move along the columns of the input feature map using 'j', and jump
      // with amount "stride"
      for (size_t j = 0; j < w_limit; j += kernel_info.stride) {
        // Ensure output indices don't exceed allocated featureMaps size
        if (out_FM_height_Iter >= featureMaps[d].size() ||
            out_FM_width_Iter >= featureMaps[d][out_FM_height_Iter].size()) {
          continue; // Skip out of bounds writes
        }

        // choose the method of pooling based on the pooling type
        switch (poolingType) {
        case maxPooling: {
          // max pooling works by comparing each entry of the overlapped part of
          // the filter and the input feature map and take the max and put it in
          // the output feature map
          double max = -1e9; // Initialize to very small number
          bool initialized = false;

          for (size_t k1 = i; k1 < (i + kernel_info.filter_height); k1++) {
            for (size_t k2 = j; k2 < (j + kernel_info.filter_width); k2++) {
              if (k1 < inputFeatureMaps[d].size() &&
                  k2 < inputFeatureMaps[d][k1].size()) {
                double val = inputFeatureMaps[d][k1][k2];
                if (!initialized || val > max) {
                  max = val;
                  initialized = true;
                }
              }
            }
          }

          if (initialized)
            featureMaps[d][out_FM_height_Iter][out_FM_width_Iter] = max;
          else
            featureMaps[d][out_FM_height_Iter][out_FM_width_Iter] = 0.0;

          break;
        }
        case averagePooling: {
          // average pooling works by averaging each entry of the overlapped
          // part of the filter and  the input feature map
          double average = 0.0;
          int count = 0;
          for (size_t k1 = i; k1 < (i + kernel_info.filter_height); k1++) {
            for (size_t k2 = j; k2 < (j + kernel_info.filter_width); k2++) {
              if (k1 < inputFeatureMaps[d].size() &&
                  k2 < inputFeatureMaps[d][k1].size()) {
                average += inputFeatureMaps[d][k1][k2];
                count++;
              }
            }
          }

          // average /=
          // static_cast<double>(kernel_info.filter_height*kernel_info.filter_width);
          // Better to average over valid pixels only if padded? But here we do
          // strict? Sticking to original logic: valid theoretical area
          if (count > 0)
            average /= static_cast<double>(kernel_info.filter_height *
                                           kernel_info.filter_width);

          featureMaps[d][out_FM_height_Iter][out_FM_width_Iter] = average;
          break;
        }
        }

        // increment the width(column) iterator
        out_FM_width_Iter++;
      }
      // increment the height(row) iterator
      out_FM_height_Iter++;
      // reset the width (column) iterator
      out_FM_width_Iter = 0;
    }
    // reset the height(row) iterator
    out_FM_height_Iter = 0;
  }
}

void poolingLayer::backwardProp(vector<featureMapType> &inputFeatureMaps,
                                vector<featureMapType> &thisLayerGrad) {

  // Reset Gradients to 0 (Critical for accumulation)
  for (auto &map : prevLayerGrad)
    for (auto &row : map)
      fill(row.begin(), row.end(), 0.0);

// Iterate over the Output Gradients (thisLayerGrad)
// We map from Output -> Input
#pragma omp parallel for
  for (int d = 0; d < kernel_info.filter_depth; d++) {

    // Iterators for the input layer (re-calculating position based on stride)
    size_t input_row = 0;

    for (size_t i = 0; i < thisLayerGrad[d].size(); i++) { // Output Rows
      size_t input_col = 0;

      for (size_t j = 0; j < thisLayerGrad[d][i].size(); j++) { // Output Cols

        // Calculate the starting corner in the INPUT map
        // This corresponds to the loop logic in forwardProp: i * stride
        size_t start_r = i * kernel_info.stride;
        size_t start_c = j * kernel_info.stride;

        double currentGrad = thisLayerGrad[d][i][j];

        if (poolingType == maxPooling) {
          // MAX POOLING: Find the max index again and pass gradient only to it
          double maxVal = -1e9; // Start very low
          size_t max_r = start_r;
          size_t max_c = start_c;
          bool found = false;

          for (size_t k1 = start_r; k1 < (start_r + kernel_info.filter_height);
               k1++) {
            for (size_t k2 = start_c; k2 < (start_c + kernel_info.filter_width);
                 k2++) {
              // Boundary check
              if (k1 < inputFeatureMaps[d].size() &&
                  k2 < inputFeatureMaps[d][k1].size()) {
                if (inputFeatureMaps[d][k1][k2] > maxVal) {
                  maxVal = inputFeatureMaps[d][k1][k2];
                  max_r = k1;
                  max_c = k2;
                  found = true;
                }
              }
            }
          }

          if (found) {
#pragma omp atomic
            prevLayerGrad[d][max_r][max_c] += currentGrad;
          }

        } else if (poolingType == averagePooling) {
          // AVERAGE POOLING: Distribute gradient equally
          double area = static_cast<double>(kernel_info.filter_height *
                                            kernel_info.filter_width);
          double distributedGrad = currentGrad / area;

          for (size_t k1 = start_r; k1 < (start_r + kernel_info.filter_height);
               k1++) {
            for (size_t k2 = start_c; k2 < (start_c + kernel_info.filter_width);
                 k2++) {
              if (k1 < inputFeatureMaps[d].size() &&
                  k2 < inputFeatureMaps[d][k1].size()) {
#pragma omp atomic
                prevLayerGrad[d][k1][k2] += distributedGrad;
              }
            }
          }
        }
      }
    }
  }
}

void poolingLayer::backwardProp_batch(vector<featureMapType> &inputFeatureMaps,
                                      vector<featureMapType> &thisLayerGrad) {
  backwardProp(inputFeatureMaps, thisLayerGrad);
}
