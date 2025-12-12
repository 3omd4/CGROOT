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
  d_kernels.resize(kernelConfig.numOfKerenels); // Resize gradients vector
  for (size_t i = 0; i < kernelConfig.numOfKerenels; i++) {
    kernels.emplace_back(initKernel(kernelConfig, initFunc, distType));

    // Initialize Optimizers for this kernel (3D)
    // One optimizer per "row" (vector<double>) in the kernel
    kernelOptimizers[i].resize(kernelConfig.kernel_depth);
    d_kernels[i].resize(kernelConfig.kernel_depth);    // Resize gradient structure for this kernel
    for (size_t d = 0; d < kernelConfig.kernel_depth; d++) {
      kernelOptimizers[i][d].resize(kernelConfig.kernel_height); // Resize optimizer depth
      d_kernels[i][d].resize(kernelConfig.kernel_height); // Resize gradient depth
      for (size_t h = 0; h < kernelConfig.kernel_height; h++) {
        kernelOptimizers[i][d][h] = createOptimizer(optConfig); // Create optimizer for this row
        d_kernels[i][d][h].assign(kernelConfig.kernel_width, 0.0); // Init gradients to 0.0
      }
    }
  }

  // iterate and make each each feature map
  for (size_t i = 0; i < fm.FM_depth; i++) {
    featureMapType map(fm.FM_height); // make a feature map
    for (size_t j = 0; j < fm.FM_height; j++) {
      map[j].assign(fm.FM_width, 0.0); // make and initialzie each row to zeros
    }
    featureMaps.emplace_back(map); // store the feature map in the feature maps vector
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
  return k;  // return the initialized kernel
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

void convLayer::backwardProp(vector<featureMapType> &inputFeatureMaps,
                             vector<featureMapType> &thisLayerGrad){
  // Apply Activation Derivative to incoming gradients
  // Mutates thisLayerGrad in place to become dZ
  #pragma omp parallel for collapse(3)
  for (size_t d = 0; d < fm.FM_depth; d++) {
    for (size_t h = 0; h < fm.FM_height; h++) {
      for (size_t w = 0; w < fm.FM_width; w++) {
        double derivative = 0.0;
        // Use the output of this layer (featureMaps) to calculate derivative
        switch (act_Funct) {
        case RelU:
          derivative = d_reLU_Funct(featureMaps[d][h][w]);
          break;
        case Sigmoid:
          derivative = d_sigmoid_Funct(featureMaps[d][h][w]);
          break;
        case Tanh:
          derivative = d_tanh_Funct(featureMaps[d][h][w]);
          break;
        }
        thisLayerGrad[d][h][w] *= derivative;
      }
    }
  }                        
  
  // Calculate Weight Gradients (dW) and Input Gradients (dX)
  int padH = kernel_info.kernel_height - 1;
  int padW = kernel_info.kernel_width - 1;

  #pragma omp parallel for collapse(2)
    for (int k = 0; k < (int)kernel_info.numOfKerenels; k++) {
      for (int d = 0; d < (int)kernel_info.kernel_depth; d++) {
        
        // A. Calculate Weight Gradients (Convolution of Input * dZ)
        for (size_t r = 0; r < kernel_info.kernel_height; r++) {
          for (size_t c = 0; c < kernel_info.kernel_width; c++) {
            double sum = 0.0;
            for (size_t i = 0; i < fm.FM_height; i++) {
              for (size_t j = 0; j < fm.FM_width; j++) {
                // Valid convolution
                sum += inputFeatureMaps[d][i + r][j + c] * thisLayerGrad[k][i][j];
              }
            }
            d_kernels[k][d][r][c] = sum; // Overwrite for SGD
          }
        }
        // B. Calculate Previous Layer Gradients (Full Convolution of dZ * Rotated_Kernel)
        for (int r = 0; r < (int)inputFeatureMaps[0].size(); r++) {
          for (int c = 0; c < (int)inputFeatureMaps[0][0].size(); c++) {
              double sum = 0.0;
              // Convolve dZ (padded) with flipped kernel
              for (int kr = 0; kr < (int)kernel_info.kernel_height; kr++) {
                  for (int kc = 0; kc < (int)kernel_info.kernel_width; kc++) {
                      int r_grad = r - (padH - kr);
                      int c_grad = c - (padW - kc);

                      if (r_grad >= 0 && r_grad < (int)fm.FM_height &&
                          c_grad >= 0 && c_grad < (int)fm.FM_width) {
                          // Use rotated kernel weights
                          sum += thisLayerGrad[k][r_grad][c_grad] * kernels[k][d][kernel_info.kernel_height - 1 - kr][kernel_info.kernel_width - 1 - kc];
                      }
                  }
              }
              #pragma omp atomic
              prevLayerGrad[d][r][c] += sum;
          }
        }
      }
    }
}


void convLayer::backwardProp_batch(vector<featureMapType> &inputFeatureMaps,
                                   vector<featureMapType> &thisLayerGrad) {
  // 2Apply Activation Derivative
#pragma omp parallel for collapse(3)
  for (size_t d = 0; d < fm.FM_depth; d++) {
    for (size_t h = 0; h < fm.FM_height; h++) {
      for (size_t w = 0; w < fm.FM_width; w++) {
        double derivative = 0.0;
        switch (act_Funct) {
        case RelU:
          derivative = d_reLU_Funct(featureMaps[d][h][w]);
          break;
        case Sigmoid:
          derivative = d_sigmoid_Funct(featureMaps[d][h][w]);
          break;
        case Tanh:
          derivative = d_tanh_Funct(featureMaps[d][h][w]);
          break;
        }
        thisLayerGrad[d][h][w] *= derivative;
      }
    }
  }

  int padH = kernel_info.kernel_height - 1;
  int padW = kernel_info.kernel_width - 1;

#pragma omp parallel for collapse(2)
  for (int k = 0; k < (int)kernel_info.numOfKerenels; k++) {
    for (int d = 0; d < (int)kernel_info.kernel_depth; d++) {
      
      // A. Calculate Weight Gradients (Accumulate += for Batch)
      for (size_t r = 0; r < kernel_info.kernel_height; r++) {
        for (size_t c = 0; c < kernel_info.kernel_width; c++) {
          double sum = 0.0;
          for (size_t i = 0; i < fm.FM_height; i++) {
            for (size_t j = 0; j < fm.FM_width; j++) {
               sum += inputFeatureMaps[d][i + r][j + c] * thisLayerGrad[k][i][j];
            }
          }
          // Atomic accumulation for batch gradient
          #pragma omp atomic
          d_kernels[k][d][r][c] += sum;
        }
      }

      // B. Calculate Previous Layer Gradients (Same as Single)
      for (int r = 0; r < (int)inputFeatureMaps[0].size(); r++) {
        for (int c = 0; c < (int)inputFeatureMaps[0][0].size(); c++) {
            double sum = 0.0;
            for (int kr = 0; kr < (int)kernel_info.kernel_height; kr++) {
                for (int kc = 0; kc < (int)kernel_info.kernel_width; kc++) {
                    int r_grad = r - (padH - kr);
                    int c_grad = c - (padW - kc);

                    if (r_grad >= 0 && r_grad < (int)fm.FM_height &&
                        c_grad >= 0 && c_grad < (int)fm.FM_width) {
                        sum += thisLayerGrad[k][r_grad][c_grad] * kernels[k][d][kernel_info.kernel_height - 1 - kr][kernel_info.kernel_width - 1 - kc];
                    }
                }
            }
            #pragma omp atomic
            prevLayerGrad[d][r][c] += sum;
        }
      }
    }
  }
}