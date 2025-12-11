#include "../Initialization/initialization.h"
#include "layers.h"

// Destructor to clean up optimizers
convLayer::~convLayer() {
  for (auto &kernel_opt : kernelOptimizers) {
    for (auto &depth_opt : kernel_opt) {
      for (auto &row_opt : depth_opt) {
        delete row_opt;
      }
    }
  }
  kernelOptimizers.clear();
}

convLayer::convLayer(convKernels &kernelConfig, activationFunction actFunc,
                     initFunctions initFunc, distributionType distType,
                     featureMapDim &FM_Dim, OptimizerConfig optConfig)
    : kernel_info(kernelConfig), fm(FM_Dim), act_Funct(actFunc) {

  // make and initialize each kernel and store them in kernels vector
  kernelOptimizers.resize(kernelConfig.numOfKerenels);
  for (size_t i = 0; i < kernelConfig.numOfKerenels; i++) {
    kernels.emplace_back(initKernel(kernelConfig, initFunc, distType));

    // Initialize Optimizers for this kernel (3D)
    // One optimizer per "row" (vector<double>) in the kernel
    kernelOptimizers[i].resize(kernelConfig.kernel_depth);
    for (size_t d = 0; d < kernelConfig.kernel_depth; d++) {
      kernelOptimizers[i][d].resize(kernelConfig.kernel_height);
      for (size_t h = 0; h < kernelConfig.kernel_height; h++) {
        kernelOptimizers[i][d][h] = createOptimizer(optConfig);
      }
    }
  }

  // iterate and make each each feature map
  for (size_t i = 0; i < fm.FM_depth; i++) {
    featureMapType map(fm.FM_height); // make a feature map
    for (size_t j = 0; j < fm.FM_height; j++) {
      map[j].assign(fm.FM_width, 0.0); // make and initialzie each row to zeros
    }
    featureMaps.emplace_back(
        map); // store the feature map in the feature maps vector
  }
}

// initialize a kernel
convLayer::kernelType convLayer::initKernel(convKernels &kernelConfig,
                                            initFunctions initFunc,
                                            distributionType distType) {
  // make a 3D kernel
  kernelType k(kernelConfig.kernel_depth); // make a vector of 2D kernels

  // iterate over each 2D kernel
  for (size_t i = 0; i < k.size(); i++) {
    // make the 2D kernel
    k[i].resize(kernelConfig.kernel_height);

    // iterate over each row of the kernel
    for (size_t j = 0; j < k[i].size(); j++) {
      // make the row of the kernel and initialize it to zeros, which will be
      // later initialized to random variables
      k[i][j].assign(kernelConfig.kernel_width, 0.0);

      // calculate n_in, the number of inputs
      size_t n_in = kernelConfig.kernel_depth * kernelConfig.kernel_height *
                    kernelConfig.kernel_width;
      switch (initFunc) // choose which initialization function to use
      {
      case Kaiming:
        init_Kaiming(k[i][j], n_in, distType);
        break;
      case Xavier:
        size_t n_out = kernelConfig.numOfKerenels * kernelConfig.kernel_height *
                       kernelConfig.kernel_width;
        init_Xavier(k[i][j], n_in, n_out, distType);
        break;
      }
    }
  }

  // return the initialized kernel
  return k;
}

void convLayer::convolute(vector<featureMapType> &inputFeatureMaps) {
// Parallelize the outer loops (Kernels and Depth)
#pragma omp parallel for collapse(2)
  for (int krnl = 0; krnl < (int)kernel_info.numOfKerenels; krnl++) {
    for (int d = 0; d < (int)kernel_info.kernel_depth; d++) {
      for (size_t i = 0;
           i < (inputFeatureMaps[d].size() - kernel_info.kernel_height + 1);
           i++) {
        for (size_t j = 0;
             j < (inputFeatureMaps[d][i].size() - kernel_info.kernel_width + 1);
             j++) {
          for (size_t k1 = i; k1 < (i + kernel_info.kernel_height); k1++) {
            for (size_t k2 = j; k2 < (j + kernel_info.kernel_width); k2++) {
              featureMaps[krnl][i][j] += inputFeatureMaps[d][k1][k2] *
                                         kernels[krnl][d][k1 - i][k2 - j];
            }
          }
        }
      }
    }
  }
}

void convLayer::forwardProp(vector<featureMapType> &inputFeatureMaps) {
  convolute(inputFeatureMaps);
  for (size_t d = 0; d < fm.FM_depth; d++) {
    for (size_t h = 0; h < fm.FM_height; h++) {
      for (size_t w = 0; w < fm.FM_width; w++) {
        switch (act_Funct) {
        case RelU:
          reLU_Funct(featureMaps[d][h][w]);
          break;
        case Sigmoid:
          sigmoid_Funct(featureMaps[d][h][w]);
          break;
        case Tanh:
          tanh_Funct(featureMaps[d][h][w]);
          break;
        }
      }
    }
  }
}