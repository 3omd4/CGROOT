#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <stdexcept>

#include "core/definitions.h"
#include "core/model.h"
#include "core/utils/mnist_loader.h"

namespace py = pybind11;

// Helper to convert std::vector<uint8_t> from MNISTImage to Python buffer
// protocol if needed Or just let pybind11 convert to list.

void bind_definitions(py::module &m) {
  py::enum_<OptimizerType>(m, "OptimizerType")
      .value("SGD", opt_SGD)
      .value("Adam", opt_Adam)
      .value("RMSprop", opt_RMSprop);

  py::enum_<distributionType>(m, "distributionType")
      .value("normalDistribution", normalDistribution)
      .value("uniformDistribution", uniformDistribution);

  py::enum_<activationFunction>(m, "activationFunction")
      .value("RelU", RelU)
      .value("Sigmoid", Sigmoid)
      .value("Tanh", Tanh)
      .value("Softmax", Softmax);

  py::enum_<initFunctions>(m, "initFunctions")
      .value("Xavier", Xavier)
      .value("Kaiming", Kaiming);

  py::enum_<poolingLayerType>(m, "poolingLayerType")
      .value("maxPooling", maxPooling)
      .value("averagePooling", averagePooling);

  py::class_<convKernels>(m, "convKernels")
      .def(py::init<>())
      .def_readwrite("numOfKerenels", &convKernels::numOfKerenels)
      .def_readwrite("kernel_width", &convKernels::kernel_width)
      .def_readwrite("kernel_height", &convKernels::kernel_height)
      .def_readwrite("kernel_depth", &convKernels::kernel_depth);

  py::class_<poolKernel>(m, "poolKernel")
      .def(py::init<>())
      .def_readwrite("filter_width", &poolKernel::filter_width)
      .def_readwrite("filter_height", &poolKernel::filter_height)
      .def_readwrite("filter_depth", &poolKernel::filter_depth)
      .def_readwrite("stride", &poolKernel::stride);

  py::class_<architecture>(m, "architecture")
      .def(py::init<>())
      .def_readwrite("numOfConvLayers", &architecture::numOfConvLayers)
      .def_readwrite("numOfFCLayers", &architecture::numOfFCLayers)
      .def_readwrite("kernelsPerconvLayers",
                     &architecture::kernelsPerconvLayers)
      .def_readwrite("neuronsPerFCLayer", &architecture::neuronsPerFCLayer)
      .def_readwrite("convLayerActivationFunc",
                     &architecture::convLayerActivationFunc)
      .def_readwrite("FCLayerActivationFunc",
                     &architecture::FCLayerActivationFunc)
      .def_readwrite("convInitFunctionsType",
                     &architecture::convInitFunctionsType)
      .def_readwrite("FCInitFunctionsType", &architecture::FCInitFunctionsType)
      .def_readwrite("distType", &architecture::distType)
      .def_readwrite("poolingLayersInterval",
                     &architecture::poolingLayersInterval)
      .def_readwrite("poolingtype", &architecture::poolingtype)
      .def_readwrite("kernelsPerPoolingLayer",
                     &architecture::kernelsPerPoolingLayer)
      .def_readwrite("optConfig", &architecture::optConfig);

  // Bind OptimizerConfig
  py::class_<OptimizerConfig>(m, "OptimizerConfig")
      .def(py::init<>())
      .def_readwrite("type", &OptimizerConfig::type)
      .def_readwrite("learningRate", &OptimizerConfig::learningRate)
      .def_readwrite("momentum", &OptimizerConfig::momentum)
      .def_readwrite("weightDecay", &OptimizerConfig::weightDecay)
      .def_readwrite("beta1", &OptimizerConfig::beta1)
      .def_readwrite("beta2", &OptimizerConfig::beta2)
      .def_readwrite("epsilon", &OptimizerConfig::epsilon);

  // Bind TrainingConfig
  py::class_<TrainingConfig>(m, "TrainingConfig")
      .def(py::init<>())
      .def_readwrite("epochs", &TrainingConfig::epochs)
      .def_readwrite("batch_size", &TrainingConfig::batch_size)
      .def_readwrite("validation_split", &TrainingConfig::validation_split)
      .def_readwrite("use_validation", &TrainingConfig::use_validation)
      .def_readwrite("shuffle", &TrainingConfig::shuffle)
      .def_readwrite("random_seed", &TrainingConfig::random_seed);

  // Bind TrainingMetrics
  py::class_<TrainingMetrics>(m, "TrainingMetrics")
      .def(py::init<>())
      .def_readwrite("epoch", &TrainingMetrics::epoch)
      .def_readwrite("train_loss", &TrainingMetrics::train_loss)
      .def_readwrite("train_accuracy", &TrainingMetrics::train_accuracy)
      .def_readwrite("val_loss", &TrainingMetrics::val_loss)
      .def_readwrite("val_accuracy", &TrainingMetrics::val_accuracy);
}

void bind_mnist_loader(py::module &m) {
  // We bind the data structures.
  // Since 'image' is vector<vector<vector<unsigned char>>>, pybind11 handles
  // this automatically via stl.h, providing list of lists.

  // Bind MNISTImage
  py::class_<cgroot::data::MNISTLoader::MNISTImage>(m, "MNISTImage")
      .def(py::init<>())
      .def_readwrite("pixels", &cgroot::data::MNISTLoader::MNISTImage::pixels)
      .def_readwrite("label", &cgroot::data::MNISTLoader::MNISTImage::label);

  // Bind MNISTDataset
  py::class_<cgroot::data::MNISTLoader::MNISTDataset>(m, "MNISTDataset")
      .def(py::init<>())
      .def_readwrite("images", &cgroot::data::MNISTLoader::MNISTDataset::images)
      .def_readwrite("num_images",
                     &cgroot::data::MNISTLoader::MNISTDataset::num_images)
      .def_readwrite("image_width",
                     &cgroot::data::MNISTLoader::MNISTDataset::image_width)
      .def_readwrite("image_height",
                     &cgroot::data::MNISTLoader::MNISTDataset::image_height);

  // Bind MNISTLoader static methods
  py::class_<cgroot::data::MNISTLoader>(m, "MNISTLoader")
      .def_static("load_training_data",
                  &cgroot::data::MNISTLoader::load_training_data)
      .def_static("load_test_data", &cgroot::data::MNISTLoader::load_test_data);
}

void bind_model(py::module &m) {
  py::class_<NNModel>(m, "NNModel")
      .def(py::init<architecture, size_t, size_t, size_t, size_t>(),
           py::arg("modelArch"), py::arg("numOfClasses"),
           py::arg("imageVerDim"), py::arg("imageHorDim"),
           py::arg("imageDepDim"))
      .def("classify", &NNModel::classify)
      .def("getProbabilities", &NNModel::getProbabilities)

      // --- FIX STARTS HERE ---
      .def(
          "train_epochs",
          [](NNModel &self,
             const cgroot::data::MNISTLoader::MNISTDataset &dataset,
             const TrainingConfig &config, py::object py_progress_callback,
             py::object py_log_callback, py::object py_stop_callback) {
            // 1. WRAP PYTHON CALLBACKS
            // We must wrap the Python objects in C++ functions that:
            // a) Acquire GIL before calling Python
            // b) Release GIL after returning to C++

            ProgressCallback progress_cpp = nullptr;
            if (!py_progress_callback.is_none()) {
              progress_cpp = [py_progress_callback](int epoch, int total,
                                                    double loss, double acc) {
                py::gil_scoped_acquire acquire; // Lock Python
                py_progress_callback(epoch, total, loss, acc);
              };
            }

            LogCallback log_cpp = nullptr;
            if (!py_log_callback.is_none()) {
              log_cpp = [py_log_callback](const std::string &msg) {
                py::gil_scoped_acquire acquire; // Lock Python
                py_log_callback(msg);
              };
            }

            // Note on Stop Flag:
            // Your C++ model expects atomic<bool>*, but Python passes a
            // function. Bridging this strictly requires changing C++
            // architecture to accept std::function<bool()>. For now, we pass
            // nullptr to prevent crashes, but this means the 'Stop' button
            // won't work unless you modify model.h/cpp to accept a
            // std::function for stopping.
            std::atomic<bool> *stop_ptr = nullptr;

            // 2. RELEASE GIL AND RUN TRAINING
            // This allows the GUI thread to run while C++ computes
            py::gil_scoped_release release;

            return self.train_epochs(dataset, config, progress_cpp, log_cpp,
                                     stop_ptr);
          },
          py::arg("dataset"), py::arg("config"),
          py::arg("progress_callback") = py::none(),
          py::arg("log_callback") = py::none(),
          py::arg("stop_flag") = py::none(),
          "Train model for multiple epochs with callbacks (GIL Released)");
  // --- FIX ENDS HERE ---
}

// Helper to validate model configuration
void validate_config_cpp(int num_fc_layers,
                         const std::vector<int> &neurons_list,
                         const architecture &arch, int img_h, int img_w,
                         int num_classes) {
  if (num_fc_layers != neurons_list.size()) {
    throw std::runtime_error("FC layer count mismatch");
  }

  if (num_fc_layers != arch.FCLayerActivationFunc.size()) {
    throw std::runtime_error("FC activation function count mismatch");
  }

  if (num_fc_layers != arch.FCInitFunctionsType.size()) {
    throw std::runtime_error("FC init function count mismatch");
  }

  for (size_t i = 0; i < neurons_list.size(); ++i) {
    if (neurons_list[i] <= 0) {
      throw std::runtime_error("All neuron counts must be positive.");
    }
  }

  if (img_h <= 0 || img_w <= 0) {
    throw std::runtime_error("Image dimensions must be positive.");
  }

  if (num_classes <= 0) {
    throw std::runtime_error("Number of classes must be positive.");
  }
}

// Factory function to create NNModel from config dict
NNModel *create_model(py::dict config) {
  try {
    architecture arch;

    // Defaults
    int num_conv_layers = 0;
    if (config.contains("num_conv_layers"))
      num_conv_layers = config["num_conv_layers"].cast<int>();

    int num_fc_layers = 2;
    if (config.contains("num_fc_layers"))
      num_fc_layers = config["num_fc_layers"].cast<int>();

    std::vector<int> neurons_list = {128, 10};
    if (config.contains("neurons_per_fc_layer"))
      neurons_list = config["neurons_per_fc_layer"].cast<std::vector<int>>();

    int num_classes = 10;
    if (config.contains("num_classes"))
      num_classes = config["num_classes"].cast<int>();

    int img_h = 28;
    if (config.contains("image_height"))
      img_h = config["image_height"].cast<int>();

    int img_w = 28;
    if (config.contains("image_width"))
      img_w = config["image_width"].cast<int>();

    // Use std::vector<size_t> for internal storage as per struct definition
    std::vector<size_t> neurons_list_sz;
    for (int n : neurons_list)
      neurons_list_sz.push_back((size_t)n);

    arch.numOfConvLayers = num_conv_layers;
    arch.numOfFCLayers = num_fc_layers;
    arch.neuronsPerFCLayer = neurons_list_sz;

    // --- NEW: CNN Configuration Parsing ---

    // 1. Kernels per layer
    std::vector<int> kernels_list;
    if (config.contains("kernels_per_layer")) {
      try {
        kernels_list = config["kernels_per_layer"].cast<std::vector<int>>();
      } catch (...) {
      }
    }

    // 2. Kernel Dimensions (list of tuples (h, w))
    std::vector<std::pair<int, int>> kernel_dims_list;
    if (config.contains("kernel_dims")) {
      try {
        kernel_dims_list =
            config["kernel_dims"].cast<std::vector<std::pair<int, int>>>();
      } catch (...) {
      }
    }

    // 3. Pooling Intervals
    std::vector<int> loop_intervals_list;
    if (config.contains("pooling_intervals")) {
      try {
        loop_intervals_list =
            config["pooling_intervals"].cast<std::vector<int>>();
      } catch (...) {
      }
    }

    // 4. Pooling Type
    std::string pooling_type_str = "Max";
    if (config.contains("pooling_type")) {
      pooling_type_str = config["pooling_type"].cast<std::string>();
    }
    poolingLayerType pool_type = (pooling_type_str == "Average")
                                     ? poolingLayerType::averagePooling
                                     : poolingLayerType::maxPooling;

    // Populate Architecture with parsed CNN data

    // A. Conv Kernels

    size_t current_depth = 1; // Start with image depth (grayscale)

    for (int i = 0; i < num_conv_layers; ++i) {
      convKernels ck;

      // Kernel Count
      if (i < kernels_list.size()) {
        ck.numOfKerenels = kernels_list[i];
      } else {
        ck.numOfKerenels = 6; // Default
      }

      // Kernel Dims
      if (i < kernel_dims_list.size()) {
        ck.kernel_height = kernel_dims_list[i].first;
        ck.kernel_width = kernel_dims_list[i].second;
      } else {
        ck.kernel_height = 5; // Default
        ck.kernel_width = 5;
      }

      // CRITICAL FIX: Set kernel depth to match input depth (previous layer
      // output)
      ck.kernel_depth = current_depth;

      // Update current depth for NEXT layer
      current_depth = ck.numOfKerenels;

      arch.kernelsPerconvLayers.push_back(ck);

      // Activation & Init (defaults)
      arch.convLayerActivationFunc.push_back(activationFunction::RelU);
      arch.convInitFunctionsType.push_back(initFunctions::Xavier);
    }

    // B. Pooling Layers

    for (size_t interval : loop_intervals_list) {
      arch.poolingLayersInterval.push_back(interval);
      arch.poolingtype.push_back(pool_type);

      // Default pooling kernel
      poolKernel pk;
      pk.filter_height = 2;
      pk.filter_width = 2;
      pk.stride = 2;
      pk.filter_depth = 1;
      arch.kernelsPerPoolingLayer.push_back(pk);
    }

    // --- END CNN Configuration Parsing ---

    // Defaults for FC layers
    for (int i = 0; i < num_fc_layers; ++i) {
      arch.FCLayerActivationFunc.push_back(activationFunction::RelU);
      arch.FCInitFunctionsType.push_back(initFunctions::Xavier);
    }

    arch.distType = distributionType::normalDistribution;

    // Optimizer Config
    std::string opt_type = "Adam";
    if (config.contains("optimizer"))
      opt_type = config["optimizer"].cast<std::string>();

    float lr = 0.001f;
    if (config.contains("learning_rate"))
      lr = config["learning_rate"].cast<float>();

    float decay = 0.0001f;
    if (config.contains("weight_decay"))
      decay = config["weight_decay"].cast<float>();

    float momentum = 0.9f;
    if (config.contains("momentum"))
      momentum = config["momentum"].cast<float>();

    arch.optConfig.learningRate = lr;
    arch.optConfig.weightDecay = decay;
    arch.optConfig.momentum = momentum;
    arch.optConfig.beta1 = 0.9;
    arch.optConfig.beta2 = 0.999;
    arch.optConfig.epsilon = 1e-8;

    if (opt_type == "Adam")
      arch.optConfig.type = OptimizerType::opt_Adam;
    else if (opt_type == "RMSprop" || opt_type == "RMSProp")
      arch.optConfig.type = OptimizerType::opt_RMSprop;
    else
      arch.optConfig.type = OptimizerType::opt_SGD;

    validate_config_cpp(num_fc_layers, neurons_list, arch, img_h, img_w,
                        num_classes);

    // Note: hardcoded depth 1 for now as per Python code
    return new NNModel(arch, num_classes, img_h, img_w, 1);

  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("C++ create_model failed: ") +
                             e.what());
  } catch (...) {
    throw std::runtime_error("C++ create_model failed with unknown error");
  }
}

// Helper to classify raw pixel data
int classify_pixels(NNModel &model, py::buffer image_buffer, int width,
                    int height, int stride) {
  py::buffer_info info = image_buffer.request();
  // We expect a flat buffer or 1D buffer from bits()
  if (info.ndim != 1) {
    // throw std::runtime_error("Expected 1D buffer of pixels");
    // Relaxed check: bits() returns void*, pybind sees it as buffer.
  }

  uint8_t *ptr = static_cast<uint8_t *>(info.ptr);

  // Convert to model's image format: vector<vector<vector<double>>>
  // Depth is 1
  image img_data;
  img_data.resize(1);
  img_data[0].resize(height);

  for (int y = 0; y < height; ++y) {
    img_data[0][y].resize(width);
    for (int x = 0; x < width; ++x) {
      int idx = y * stride + x;
      // Safety check (might slow down, can remove if trusted)
      // if (idx >= info.size) throw std::runtime_error("Buffer overflow");

      img_data[0][y][x] = static_cast<double>(ptr[idx]);
    }
  }

  return model.classify(img_data);
}

PYBIND11_MODULE(cgroot_core, m) {
  m.doc() = "CGROOT Core Neural Network Library Bindings";

  bind_definitions(m);
  bind_mnist_loader(m);
  bind_model(m);

  // Bind new factory/utility functions
  m.def("create_model", &create_model, py::return_value_policy::take_ownership,
        "Create a new NNModel from a configuration dictionary");
  m.def("classify_pixels", &classify_pixels,
        "Classify image from raw pixel buffer");
}
