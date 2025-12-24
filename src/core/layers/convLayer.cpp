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

convLayer::convLayer(convKernels &kernelConfig, activationFunction actFunc,
                     initFunctions initFunc, distributionType distType,
                     featureMapDim &inputFM_Dim, OptimizerConfig optConfig)
    : kernel_info(kernelConfig), act_Funct(actFunc) {
  // a check to make sure that the stride isn't zero so no error occur from
  // division by 0
  kernelConfig.stride = (kernelConfig.stride != 0) ? kernelConfig.stride : 1;
  kernel_info.stride = kernelConfig.stride;

  // calculate the dimensions of the output feature maps
  fm.FM_depth = kernelConfig.numOfKerenels;
  fm.FM_height = ((inputFM_Dim.FM_height - kernelConfig.kernel_height +
                   2 * kernelConfig.padding) /
                  kernelConfig.stride) +
                 1;
  fm.FM_width = ((inputFM_Dim.FM_width - kernelConfig.kernel_width +
                  2 * kernelConfig.padding) /
                 kernelConfig.stride) +
                1;

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

  // iterate and make each each feature map
  for (size_t i = 0; i < fm.FM_depth; i++) {
    featureMapType map(fm.FM_height); // make a feature map
    for (size_t j = 0; j < fm.FM_height; j++) {
      map[j].assign(fm.FM_width, 0.0); // make and initialzie each row to zeros
    }
    featureMaps.emplace_back(
        map); // store the feature map in the feature maps vector
  }

  prevLayerGrad.resize(inputFM_Dim.FM_depth);
  for (size_t i = 0; i < inputFM_Dim.FM_depth; i++) {
    prevLayerGrad[i].resize(inputFM_Dim.FM_height);
    for (size_t j = 0; j < inputFM_Dim.FM_height; j++) {
      prevLayerGrad[i][j].assign(inputFM_Dim.FM_width, 0.0);
    }
  }

  // initialze the biases
  // because of the Dying ReLU problem, if the activation function is ReLU then
  // initialize with 0.01, else initialize with zero
  if (actFunc == RelU) {
    bias.assign(kernelConfig.numOfKerenels, 0.01);
  } else {
    bias.assign(kernelConfig.numOfKerenels, 0.0);
  }

  d_bias.assign(kernelConfig.numOfKerenels, 0.0);
  biasOptimizer = createOptimizer(optConfig);
}
// initialize a kernel
// input:        -kernelConfig (contains all the information about the kernel)
//               -initFunc (the initialization function)
//               -distType (the type of the distribution)
// output:       kernelType (the initialized kernel)
// side effect:  N/A
// Note:         N/A
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
  // do the same convolution operation using every kernel where each kernel
  // will result in a different feature map, iterate using "krnl"
  for (int krnl = 0; krnl < static_cast<int>(kernel_info.numOfKerenels);
       krnl++) {

    // iterate on every channel (depth) using 'd', where 'd' is the depth
    // corrisponding of the kernel depth and the input feature map depth
    for (int d = 0; d < static_cast<int>(kernel_info.kernel_depth); d++) {
      // moving on the 2D feature map the length to be moved is fm_dim -
      // krnl_dim + 1

      int i_iter =
          -kernel_info.padding; // iterate the rows of the input feature map

      // move along the rows of the output feature map using 'i'
      for (int i = 0; i < static_cast<int>(fm.FM_height); i++) {
        // reset the output feature maps with zeros on the first channel of the
        // input feature map for each kernel
        if (d == 0) {
          fill(featureMaps[krnl][i].begin(), featureMaps[krnl][i].end(), 0.0);
        }

        int j_iter =
            -kernel_info
                 .padding; // iterate the columns of the input feature map

        // move along the columns of the ouput feature map using 'j'
        for (int j = 0; j < static_cast<int>(fm.FM_width); j++) {

          // do the convolution
          //"k1" is row iterator and "k2" is the column iterator
          for (int k1 = i_iter; k1 < (i_iter + kernel_info.kernel_height);
               k1++) {
            for (int k2 = j_iter; k2 < (j_iter + kernel_info.kernel_width);k2++)
            {
              // the convolutio is done by doing element-wise multiplication of
              // the input feature map and the kernel infront of this part of
              // the feature map this is done for every channel of the input
              // feature map and then stored in the corrisponding entry in the
              // output feature map

              double inputVal;
              if ((k1 < 0) ||
                  (k1 >= static_cast<int>(inputFeatureMaps[d].size())) ||
                  (k2 < 0) ||
                  (k2 >= static_cast<int>(inputFeatureMaps[d][k1].size()))) {
                inputVal = 0.0;
              } else {
                inputVal = inputFeatureMaps[d][k1][k2];
              }

              //"krnl" indexes which kernel and its output feature map
              //'d' is the depth of both the kernel and the input feature map
              //"k1" and "k2" are the indexs of the row and column of which the
              // the kernel is positioned, respectively, thsi position is the
              // top-left corner of the kernel the result of the differnt
              // channels of the same kernel are added to the corrisponding
              // entry in the output feature map when the depth iterator 'd' is
              // changed
              featureMaps[krnl][i][j] +=
                  inputVal * kernels[krnl][d][k1 - i_iter][k2 - j_iter];
            }
          }
          if (d == (kernel_info.kernel_depth - 1)) {
            featureMaps[krnl][i][j] += bias[krnl];
          }

          // move the kernel by the stride amount above the input feature map in
          // the horizontal direction
          j_iter += kernel_info.stride;
        }
        // move the kernel by the stride amount above the input feature map in
        // the vertical direction
        i_iter += kernel_info.stride;
      }
    }
  }
}

// do the forward propagation of the convolution layer
// by first applying the convolution and then the activation functions
// input:        inputFeatureMaps
// output:       N/A
// side effect:  the feature maps are filled with the forward propagation values
// note:         N/A
void convLayer::forwardProp(vector<featureMapType> &inputFeatureMaps) {
  // STEP 1: Apply convolution operation
  // Convolve kernels over input feature maps with stride and padding to produce raw output
  convolute(inputFeatureMaps);

  // STEP 2: Apply non-linear activation function element-wise to the output
  // This introduces non-linearity to the network and helps learn complex patterns
  // Activation function applied: ReLU, Sigmoid, or Tanh based on layer configuration
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

// BACKWARD PROPAGATION (BACKPROP) - Single Sample Mode
// This function computes gradients for weight updates and propagates error to previous layer
// Used in Stochastic Gradient Descent (SGD) - updates after each single sample
// input:                -inputFeatureMaps (cached activations from forward pass)
//                       -thisLayerGrad (gradient of loss w.r.t output of this layer)
// output:               N/A
// side effect:          1. prevLayerGrad is filled with error to propagate to previous layer
//                       2. d_kernels accumulates weight gradients (dW)
//                       3. d_bias accumulates bias gradients (dB)
// Note:                 This function works for single samples. For batch mode, use backwardProp_batch()
void convLayer::backwardProp(vector<featureMapType> &inputFeatureMaps,
                             vector<featureMapType> &thisLayerGrad) {
  // Cache dimension values for efficient lookup
  // These represent the spatial dimensions of input feature maps and kernels
  int inputH = inputFeatureMaps[0].size();
  int inputW = inputFeatureMaps[0][0].size();
  int kH = kernel_info.kernel_height;
  int kW = kernel_info.kernel_width;
  int stride = kernel_info.stride;
  int pad = kernel_info.padding;

// PHASE 1: Activation Derivative & Bias Gradients
// Chain rule: dL/dZ = dL/dA * dA/dZ (where Z is pre-activation, A is post-activation)
// This accounts for the non-linearity introduced by the activation function
#pragma omp parallel for
  for (int d = 0; d < fm.FM_depth; d++) {
    double bias_sum = 0.0; // Local variable prevents Race Condition

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
        bias_sum += thisLayerGrad[d][h][w];
      }
    }
    d_bias[d] += bias_sum;
  }

// PHASE 2: Compute Gradients for Weights (dW) and for Input (dX)
// Chain rule applications:
//   - Weight Gradient: dL/dW = dL/dZ * dZ/dW = thisLayerGrad[k][i][j] * input_pixel
//   - Input Gradient: dL/dX = sum over all output positions (dL/dZ * dZ/dX)
// Parallelize over kernels (k) and input channels (d) for better cache locality
#pragma omp parallel for collapse(2)
  for (int k = 0; k < (int)kernel_info.numOfKerenels; k++) {
    for (int d = 0; d < (int)kernel_info.kernel_depth; d++) {

      // Iterate through each position in the OUTPUT gradient feature map
      // This represents the error signal flowing back from the next layer
      for (int i = 0; i < (int)fm.FM_height; i++) {
        for (int j = 0; j < (int)fm.FM_width; j++) {
          double grad = thisLayerGrad[k][i][j];
          if (grad == 0.0)
            continue; // Skip zero gradients for efficiency

          // CRUCIAL: Map output position (i,j) back to input positions
          // For each kernel position (r,c), find which input pixel it operated on
          // This reverses the forward pass convolution operation
          for (int r = 0; r < kH; r++) {
            for (int c = 0; c < kW; c++) {

              // Convert kernel-relative coordinates to input coordinates
              // Formula: in_pos = output_pos * stride - padding + kernel_pos
              // This accounts for stride (sampling interval) and padding (border handling)
              // Must match forward pass calculation exactly to maintain correctness
              int in_r = i * stride - pad + r;
              int in_c = j * stride - pad + c;

              // Only process if input position is within valid bounds
              // (Some positions may be outside due to padding/stride)
              if (in_r >= 0 && in_r < inputH && in_c >= 0 && in_c < inputW) {

                // COMPUTATION A: Accumulate Weight Gradient (dW)
                // dW[k][d][r][c] accumulates contributions: Input_value * Error_signal
                // This tells us how much to adjust this weight to reduce error
                d_kernels[k][d][r][c] += inputFeatureMaps[d][in_r][in_c] * grad;

// COMPUTATION B: Accumulate Input Gradient (dX) - Error backpropagation to previous layer
// dX[d][in_r][in_c] = Kernel_weight * Error_signal
// Multiple output positions may map to same input pixel (overlapping receptive fields)
// Use atomic operation to safely accumulate without race conditions in parallel execution
#pragma omp atomic
                prevLayerGrad[d][in_r][in_c] += kernels[k][d][r][c] * grad;
              }
            }
          }
        }
      }
    }
  }
}
// BACKWARD PROPAGATION BATCH MODE (BACKPROP_BATCH)
// This function computes gradients for entire batch and must reset gradients
// Used in Batch Gradient Descent (BGD) - accumulates gradients over entire batch before update
// input:                -inputFeatureMaps (vector of cached activations from forward pass)
//                       -thisLayerGrad (gradient of loss w.r.t output of this layer for batch)
// output:               N/A
// side effect:          1. prevLayerGrad reset to zero, then filled with accumulated batch errors
//                       2. d_kernels reset and accumulates batch weight gradients
//                       3. d_bias reset and accumulates batch bias gradients
// Note:                 This function processes entire batch. For single samples, use backwardProp()
//                       The main difference is resetting gradient accumulators to handle batch accumulation
void convLayer::backwardProp_batch(vector<featureMapType> &inputFeatureMaps,
                                   vector<featureMapType> &thisLayerGrad) {
  // Cache dimension values as const for optimization
  // These define the spatial dimensions used in convolution calculations
  const int inputH = inputFeatureMaps[0].size();
  const int inputW = inputFeatureMaps[0][0].size();
  const int kH = kernel_info.kernel_height;
  const int kW = kernel_info.kernel_width;
  const int stride = kernel_info.stride;
  const int pad = kernel_info.padding;

  // BATCH MODE CRITICAL STEP: Reset all accumulated gradients to zero
  // This ensures we don't mix gradients from previous batch passes
  // In SGD mode (backwardProp), gradients are accumulated; here we start fresh
  for (auto &ch : prevLayerGrad)
    for (auto &row : ch)
      fill(row.begin(), row.end(), 0.0);

// PHASE 1: Compute activation derivatives and accumulate bias gradients
// Apply chain rule through activation function: dL/dZ = dL/dA * dA/dZ
// Process all samples in batch and accumulate bias updates
// Parallelized over output channels (kernels)
#pragma omp parallel for
  for (int k = 0; k < fm.FM_depth; k++) {
    double local_bias = 0.0;

    for (int i = 0; i < fm.FM_height; i++) {
      for (int j = 0; j < fm.FM_width; j++) {
        double deriv = 0.0;
        switch (act_Funct) {
        case RelU:
          deriv = d_reLU_Funct(featureMaps[k][i][j]);
          break;
        case Sigmoid:
          deriv = d_sigmoid_Funct(featureMaps[k][i][j]);
          break;
        case Tanh:
          deriv = d_tanh_Funct(featureMaps[k][i][j]);
          break;
        }
        thisLayerGrad[k][i][j] *= deriv;
        local_bias += thisLayerGrad[k][i][j];
      }
    }

#pragma omp atomic
    d_bias[k] += local_bias;
  }

// PHASE 2: Accumulate Weight Gradients (dW) and Input Gradients (dX) for entire batch
// This is the "true convolution" operation that computes gradient tensors
// Chain rule for gradients:
//   - dW[k][d][r][c] += Input[d][in_r][in_c] * Error[k][i][j]
//   - dX[d][in_r][in_c] += Kernel[k][d][r][c] * Error[k][i][j]
// Parallelize over kernels and input channels for maximum parallelism
#pragma omp parallel for collapse(2)
  for (int k = 0; k < kernel_info.numOfKerenels; k++) {
    for (int d = 0; d < kernel_info.kernel_depth; d++) {

      // Iterate through output gradient feature map (error flowing back)
      for (int i = 0; i < fm.FM_height; i++) {
        for (int j = 0; j < fm.FM_width; j++) {

          // Cache gradient value; skip if zero to improve efficiency
          const double grad = thisLayerGrad[k][i][j];
          if (grad == 0.0)
            continue;

          // Map output position back to input spatial coordinates
          // Account for kernel size, stride, and padding in the mapping
          for (int r = 0; r < kH; r++) {
            for (int c = 0; c < kW; c++) {

              // Calculate which input pixel contributed to this output position
              // Inverse of forward convolution: output = f(input with stride/padding)
              const int in_r = i * stride - pad + r;
              const int in_c = j * stride - pad + c;

              // Validate input coordinates are within bounds
              if (in_r >= 0 && in_r < inputH && in_c >= 0 && in_c < inputW) {

                // GRADIENT 1: Weight Gradient (dW) - How to adjust kernel to reduce error
                // Accumulate: dW += Input_pixel * Error_signal
                // Atomic operation prevents race conditions from parallel writes
#pragma omp atomic
                d_kernels[k][d][r][c] += inputFeatureMaps[d][in_r][in_c] * grad;

                // GRADIENT 2: Input Gradient (dX) - Error to propagate backward
                // Accumulate: dX += Kernel_weight * Error_signal
                // Atomic operation ensures thread-safe accumulation of overlapping contributions
#pragma omp atomic
                prevLayerGrad[d][in_r][in_c] += kernels[k][d][r][c] * grad;
              }
            }
          }
        }
      }
    }
  }
}

// UPDATE KERNELS - Single Sample Mode (Stochastic Gradient Descent)
// This updates all kernels and biases using accumulated gradients from one sample
// Used after backwardProp() to apply single-sample gradient updates
// input:          N/A
// ouput:          N/A
// side effect:    1. Kernels updated using optimizer (SGD, Adam, etc.) with their gradients
//                 2. Biases updated similarly
//                 3. All gradient accumulators (d_kernels, d_bias) reset to zero for next sample
// Note:           For batch mode with averaging, use update_batch() instead
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

  // update biases
  biasOptimizer->update(bias, d_bias);
  fill(d_bias.begin(), d_bias.end(), 0.0);
}

// UPDATE KERNELS - Batch Mode (Batch Gradient Descent)
// This updates all kernels and biases after processing entire batch
// CRITICAL: Averages accumulated gradients by batch size before applying updates
// Used after backwardProp_batch() to apply averaged batch gradient updates
// input:          numOfExamples (total number of samples in the batch)
// ouput:          N/A
// side effect:    1. Accumulated gradients averaged by dividing by batch size
//                 2. Kernels and biases updated using averaged gradients via optimizer
//                 3. All gradient accumulators reset to zero for next batch
// Note:           For single-sample updates, use update() instead (no averaging needed)
//                 Averaging prevents large weight updates that batch processing would cause
void convLayer::update_batch(int numOfExamples) {
  // Calculate scaling factor: 1/N where N is batch size
  // This converts accumulated gradients (sum over batch) to average gradients
  // Each gradient value is multiplied by this scale before optimizer uses it
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
  }

  // update biases
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
  kernelOptimizers.clear();
}