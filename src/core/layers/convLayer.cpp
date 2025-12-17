

#include "../Initialization/initialization.h"
#include "layers.h"

// the convolution layer constructor
// input:        -kernelConfig (contains all the information about the kernel)
//               -actFunc (activation function)
//               -initFunc (initialization function)
//               -distType (distribution type)
//               -FM_Dim (the dimension of the output feature map)
// ouput:        N/A
// side effect:  the convolution layer is constructed
// Note:         N/A
convLayer::convLayer(const convKernels &kernelConfig,
                     activationFunction actFunc, initFunctions initFunc,
                     distributionType distType, featureMapDim &FM_Dim,
                     OptimizerConfig optConfig)
    : kernel_info(kernelConfig), fm(FM_Dim), act_Funct(actFunc) {

  d_kernels.resize(kernelConfig.numOfKerenels); // Resize gradients vector
  kernelOptimizers.resize(kernelConfig.numOfKerenels);
  // make and initialize each kernel and store them in kernels vector
  for (size_t i = 0; i < kernelConfig.numOfKerenels; i++) {
    kernels.emplace_back(initKernel(kernelConfig, initFunc, distType));

    d_kernels[i].resize(
        kernelConfig.kernel_depth); // Resize gradient structure for this kernel

    kernelOptimizers[i].resize(kernelConfig.kernel_depth);
    for (size_t d = 0; d < kernelConfig.kernel_depth; d++) {

      kernelOptimizers[i][d].resize(kernelConfig.kernel_height);
      d_kernels[i][d].resize(
          kernelConfig.kernel_height); // Resize gradient depth
      for (size_t h = 0; h < kernelConfig.kernel_height; h++) {
        kernelOptimizers[i][d][h] = createOptimizer(optConfig);
        d_kernels[i][d][h].assign(kernelConfig.kernel_width,
                                  0.0); // Init gradients to 0.0
      }
    }
  }

  // Initialize biases
  bias.resize(kernel_info.numOfKerenels);
  d_bias.resize(kernel_info.numOfKerenels);
  biasOptimizer = createOptimizer(optConfig);

  for (size_t i = 0; i < kernel_info.numOfKerenels; i++) {
    bias[i] = 0.0;
    d_bias[i] = 0.0;
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

  size_t inputHeight = fm.FM_height + kernelConfig.kernel_height - 1;
  size_t inputWidth = fm.FM_width + kernelConfig.kernel_width - 1;
  size_t inputDepth = kernelConfig.kernel_depth;

  prevLayerGrad.resize(inputDepth);
  for (size_t i = 0; i < inputDepth; i++) {
    prevLayerGrad[i].resize(inputHeight);
    for (size_t j = 0; j < inputHeight; j++) {
      prevLayerGrad[i][j].assign(inputWidth, 0.0);
    }
  }
}

// initialize a kernel
// input:        -kernelConfig (contains all the information about the kernel)
//               -initFunc (the initialization function)
//               -distType (the type of the distribution)
// output:       kernelType (the initialized kernel)
// side effect:  N/A
// Note:         N/A
convLayer::kernelType convLayer::initKernel(const convKernels &kernelConfig,
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

      // calculate n_in, the number of inputs, which is used in the
      // initialization functions
      size_t n_in = kernelConfig.kernel_depth * kernelConfig.kernel_height *
                    kernelConfig.kernel_width;
      switch (initFunc) // choose which initialization function to use
      {
      case Kaiming:
        init_Kaiming(k[i][j], n_in, distType);
        break;
      case Xavier:
        // calculate n_out, the number of outputs to be used by Xavier
        // initialization function
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

// do the convolution operation by sweeping the kernels through
// the input feature map and putin the result in the (output) feature map
// essentially doing the forward propagation
// input:        inputFeatureMaps (previous layer output feature maps)
// output:       N/A
// side effect:  this layer (output) feature maps is filled
// Note:         N/A
void convLayer::convolute(vector<featureMapType> &inputFeatureMaps) {
  // 1. Validate Basic Integrity
  if (kernels.size() != kernel_info.numOfKerenels ||
      featureMaps.size() != kernel_info.numOfKerenels) {
    std::cerr << "Critical Error: Kernel/FeatureMap size mismatch" << std::endl;
    exit(1);
  }

  // 2. Initialize Output Feature Maps with Bias (Move this OUTSIDE the 'd'
  // loop) This ensures the map is cleared and ready even if input channels are
  // skipped
  for (size_t k = 0; k < kernel_info.numOfKerenels; k++) {
    for (size_t r = 0; r < featureMaps[k].size(); r++) {
      std::fill(featureMaps[k][r].begin(), featureMaps[k][r].end(), bias[k]);
    }
  }

  // 3. Perform Convolution
  for (size_t krnl = 0; krnl < kernel_info.numOfKerenels; krnl++) {
    for (size_t d = 0; d < kernel_info.kernel_depth; d++) {

      // Skip invalid inputs safely
      if (d >= inputFeatureMaps.size() || inputFeatureMaps[d].empty())
        continue;

      size_t inputH = inputFeatureMaps[d].size();
      if (inputH < kernel_info.kernel_height)
        continue;

      // Calculate output limits
      size_t h_limit = inputH - kernel_info.kernel_height + 1;
      if (h_limit > featureMaps[krnl].size())
        h_limit = featureMaps[krnl].size();

      for (size_t i = 0; i < h_limit; i++) {
        size_t inputW = inputFeatureMaps[d][i].size();
        if (inputW < kernel_info.kernel_width)
          continue;

        size_t w_limit = inputW - kernel_info.kernel_width + 1;
        if (w_limit > featureMaps[krnl][i].size())
          w_limit = featureMaps[krnl][i].size();

        for (size_t j = 0; j < w_limit; j++) {
          // Optimized inner loop
          double val = 0.0;
          for (size_t k1 = 0; k1 < kernel_info.kernel_height; k1++) {
            // Pre-calculate row index
            size_t in_row = i + k1;

            const auto &row_vec = inputFeatureMaps[d][in_row];
            const auto &kernel_row = kernels[krnl][d][k1];

            for (size_t k2 = 0; k2 < kernel_info.kernel_width; k2++) {
              // Bounds check optimization: k2+j is col index
              if ((j + k2) < row_vec.size()) {
                val += row_vec[j + k2] * kernel_row[k2];
              }
            }
          }
          featureMaps[krnl][i][j] += val;
        }
      }
    }
  }
}

// do the forward propagation of the convolution layer
// by first applying the convolution and then the activation functions
// input:        inputFeatureMaps
// output:       N/A
// side effect:  the feature maps are filled with the forward propagation
// values note:         N/A
void convLayer::forwardProp(vector<featureMapType> &inputFeatureMaps) {
  // apply the convolution
  convolute(inputFeatureMaps);

  // apply the activation function to every element of the output feature map
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

// backward propagate the error
// input:                -inputFeatureMaps
//                       -thisLayerGrad
// output:               N/A
// side effect:          the prevLayerGrad is filled with the error to be
// propagated Note:                 This function works with SGD or for
// updating after a single
void convLayer::backwardProp(vector<featureMapType> &inputFeatureMaps,
                             vector<featureMapType> &thisLayerGrad) {
  for (auto &depth : prevLayerGrad) {
    for (auto &row : depth) {
      std::fill(row.begin(), row.end(), 0.0);
    }
  }

// Apply Activation Derivative to incoming gradients
// Mutates thisLayerGrad in place to become dZ
#pragma omp parallel for collapse(3)
  for (int d = 0; d < fm.FM_depth; d++) {
    for (int h = 0; h < fm.FM_height; h++) {
      for (int w = 0; w < fm.FM_width; w++) {
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

  // Calculate Bias Gradients (dB) -> sum over height and width for each
  // kernel
  for (int k = 0; k < (int)kernel_info.numOfKerenels; k++) {
    double sum = 0.0;
    for (int d = 0; d < (int)kernel_info.kernel_depth; d++) {
      // Bias is per kernel (per output feature map), usually summed over one
      // depth channel? Wait, convLayer bias: one bias per kernel. The kernel
      // output is 2D (HxW) per kernel (since we sum over depth). The output
      // feature map 'k' has dimensions HxW. bias[k] is added to every pixel
      // in featureMaps[k]. So d_bias[k] = sum_{h,w} thisLayerGrad[k][h][w].
      // But thisLayerGrad has dimensions [depth][height][width]. Wait,
      // backwardProp inputs: `vector<featureMapType> &thisLayerGrad`.
      // `thisLayerGrad` corresponds to output of THIS layer. `featureMaps`
      // (output) has dim [numOfKernels][height][width]. (Wait, verify
      // `featureMaps` dim). In `layers.h`: `vector<featureMapType>
      // featureMaps;` In `convolute`: `featureMaps[krnl][i][j]`. So dimension
      // 0 is Kernel Index. In `backwardProp` signature: `thisLayerGrad`. In
      // `backwardProp` body using `thisLayerGrad`: Loop d over `fm.FM_depth`
      // (which should be equal to numOfKernels). So `thisLayerGrad[d]`
      // corresponds to kernel `d`.

      for (int h = 0; h < fm.FM_height; h++) {
        for (int w = 0; w < fm.FM_width; w++) {
          sum += thisLayerGrad[k][h][w];
        }
      }
    }
    // Correct logic: we iterate k=0..numOfKernels.
    // However, `thisLayerGrad` is indexed by [d][h][w] where d is
    // 0..fm.FM_depth. fm.FM_depth should match numOfKernels! In constructor:
    // `fm.FM_depth = modelArch.kernelsPerconvLayers[i].numOfKerenels;` (seen
    // in model.cpp). So yes, d in loop corresponds to k.
    d_bias[k] = sum;
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
      // B. Calculate Previous Layer Gradients (Full Convolution of dZ *
      // Rotated_Kernel)
      for (int r = 0; r < (int)inputFeatureMaps[0].size(); r++) {
        for (int c = 0; c < (int)inputFeatureMaps[0][0].size(); c++) {
          double sum = 0.0;
          // Convolve dZ (padded) with flipped kernel
          for (int kr = 0; kr < (int)kernel_info.kernel_height; kr++) {
            for (int kc = 0; kc < (int)kernel_info.kernel_width; kc++) {
              int r_grad = r - (padH - kr);
              int c_grad = c - (padW - kc);

              if (r_grad >= 0 && r_grad < (int)fm.FM_height && c_grad >= 0 &&
                  c_grad < (int)fm.FM_width) {
                // Use rotated kernel weights
                sum += thisLayerGrad[k][r_grad][c_grad] *
                       kernels[k][d][kernel_info.kernel_height - 1 - kr]
                              [kernel_info.kernel_width - 1 - kc];
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

// backward propagate the error after a batch
// input:                -inputFeatureMaps
//                       -thisLayerGrad
// output:               N/A
// side effect:          the prevLayerGrad is filled with the error to be
// propagated Note:                 This function works with BGD or for
// updating after a whole batch
void convLayer::backwardProp_batch(vector<featureMapType> &inputFeatureMaps,
                                   vector<featureMapType> &thisLayerGrad) {
  for (auto &depth : prevLayerGrad) {
    for (auto &row : depth) {
      std::fill(row.begin(), row.end(), 0.0);
    }
  }

  // 2Apply Activation Derivative
#pragma omp parallel for collapse(3)
  for (int d = 0; d < fm.FM_depth; d++) {
    for (int h = 0; h < fm.FM_height; h++) {
      for (int w = 0; w < fm.FM_width; w++) {
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

  // Calculate Bias Gradients (dB) - Accumulate for batch
  for (int k = 0; k < (int)kernel_info.numOfKerenels; k++) {
    double sum = 0.0;
    for (int d = 0; d < (int)kernel_info.kernel_depth; d++) {
      for (int h = 0; h < fm.FM_height; h++) {
        for (int w = 0; w < fm.FM_width; w++) {
          sum += thisLayerGrad[k][h][w];
        }
      }
    }
#pragma omp atomic
    d_bias[k] += sum;
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

              if (r_grad >= 0 && r_grad < (int)fm.FM_height && c_grad >= 0 &&
                  c_grad < (int)fm.FM_width) {
                sum += thisLayerGrad[k][r_grad][c_grad] *
                       kernels[k][d][kernel_info.kernel_height - 1 - kr]
                              [kernel_info.kernel_width - 1 - kc];
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

// update the kernels
// input:          N/A
// ouput:          N/A
// side effect:    the kernel is updated with new values and the kernel
// gradients are reseted Note:           This function works for single
// samples, for a batch, use upadate_batch()
void convLayer::update() {
#pragma omp parallel for
  for (int krnl = 0; krnl < static_cast<int>(kernels.size()); krnl++) {
    for (int d = 0; d < static_cast<int>(kernels[krnl].size()); d++) {
      for (int i = 0; i < static_cast<int>(kernels[krnl][d].size()); i++) {
        // update kernels
        kernelOptimizers[krnl][d][i]->update(kernels[krnl][d][i],
                                             d_kernels[krnl][d][i]);

        // reset gradients
        fill(d_kernels[krnl][d][i].begin(), d_kernels[krnl][d][i].end(), 0.0);
      }
    }
  }
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

// update the kernels
// input:          numOfExamples
// ouput:          N/A
// side effect:    the kernel is updated with new values and the kernel
// gradients are reseted Note:           This function works for a batch of
// samples, for a single sample, use upadate()
void convLayer::update_batch(int numOfExamples) {
  double scale = 1.0 / static_cast<double>(numOfExamples);

#pragma omp parallel for
  for (int krnl = 0; krnl < static_cast<int>(kernels.size()); krnl++) {
    for (int d = 0; d < static_cast<int>(kernels[krnl].size()); d++) {
      for (int i = 0; i < static_cast<int>(kernels[krnl][d].size()); i++) {

        // average the gradients
        for (int j = 0; j < static_cast<int>(kernels[krnl][d][i].size()); j++) {
          d_kernels[krnl][d][i][j] *= scale;
        }

        // update kernels
        kernelOptimizers[krnl][d][i]->update(kernels[krnl][d][i],
                                             d_kernels[krnl][d][i]);
        // reset gradients
        fill(d_kernels[krnl][d][i].begin(), d_kernels[krnl][d][i].end(), 0.0);
      }
    }
    // Update bias
    d_bias[krnl] *= scale;
  }
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

convLayer::~convLayer() {
  for (auto &kernel_opt : kernelOptimizers) {
    for (auto &depth_opt : kernel_opt) {
      for (auto &row_opt : depth_opt) {
        delete row_opt;
      }
    }
  }
  delete biasOptimizer;
  kernelOptimizers.clear();
}