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
      .value("SGD_Momentum", opt_SGD_Momentum)
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
      .def_readwrite("beta", &OptimizerConfig::beta)
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
                     &cgroot::data::MNISTLoader::MNISTDataset::image_height)
      .def_readwrite("depth", &cgroot::data::MNISTLoader::MNISTDataset::depth);

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
      .def("getLayerFeatureMaps", &NNModel::getLayerFeatureMaps)
      .def("getLayerType", &NNModel::getLayerType)
      .def("store", &NNModel::store)
      .def("load", &NNModel::load)
      .def("getTrainingHistory", &NNModel::getTrainingHistory)

      .def("getInputHeight", &NNModel::getInputHeight)
      .def("getInputWidth", &NNModel::getInputWidth)
      .def("getInputDepth", &NNModel::getInputDepth)
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
                                                    double loss, double acc,
                                                    int current_image_idx) {
                py::gil_scoped_acquire acquire; // Lock Python
                py_progress_callback(epoch, total, loss, acc,
                                     current_image_idx);
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
          "Train model for multiple epochs with callbacks (GIL Released)")

      .def(
          "evaluate",
          [](NNModel &self,
             const cgroot::data::MNISTLoader::MNISTDataset &dataset,
             py::object py_progress_callback) {
            ProgressCallback progress_cpp = nullptr;
            if (!py_progress_callback.is_none()) {
              progress_cpp = [py_progress_callback](int epoch, int total,
                                                    double loss, double acc,
                                                    int current_image_idx) {
                py::gil_scoped_acquire acquire; // Lock Python
                py_progress_callback(epoch, total, loss, acc,
                                     current_image_idx);
              };
            }

            // Release GIL for computation
            py::gil_scoped_release release;
            return self.evaluate(dataset, progress_cpp);
          },
          py::arg("dataset"), py::arg("progress_callback") = py::none(),
          "Evaluate model on dataset (GIL Released). Returns (loss, accuracy, "
          "confusion_matrix)");
  // --- FIX ENDS HERE ---
}

// Helper to map activation function string to enum
activationFunction map_activation(const std::string &name) {
  if (name == "ReLU")
    return activationFunction::RelU;
  if (name == "Sigmoid")
    return activationFunction::Sigmoid;
  if (name == "Tanh")
    return activationFunction::Tanh;
  if (name == "Softmax")
    return activationFunction::Softmax;
  if (name == "LeakyReLU")
    return activationFunction::RelU; // Map to ReLU for now
  if (name == "Linear")
    return activationFunction::RelU; // Map to ReLU for now
  return activationFunction::RelU;   // Default
}

// Helper to map init function string to enum
initFunctions map_init_function(const std::string &name) {
  if (name == "Xavier")
    return initFunctions::Xavier;
  if (name == "He" || name == "Kaiming")
    return initFunctions::Kaiming;
  return initFunctions::Kaiming; // Default to He
}

// Helper to map distribution type string to enum
distributionType map_distribution(const std::string &name) {
  if (name == "Uniform")
    return distributionType::uniformDistribution;
  return distributionType::normalDistribution; // Default
}

// Helper to resolve "Auto" initialization based on activation
std::string resolve_auto_init(const std::string &init_func,
                              const std::string &activation,
                              std::string &log_msg) {
  if (init_func != "Auto") {
    return init_func; // User override
  }

  // Auto resolution
  if (activation == "ReLU" || activation == "LeakyReLU") {
    log_msg = "Auto -> He (activation: " + activation + ")";
    return "He";
  } else { // Tanh, Sigmoid, Softmax, others
    log_msg = "Auto -> Xavier (activation: " + activation + ")";
    return "Xavier";
  }
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
    // Log received configuration for debugging
    std::cout << "\n=== C++ create_model: Received Configuration ==="
              << std::endl;

    architecture arch;

    // Defaults
    int num_conv_layers = 0;
    if (config.contains("num_conv_layers")) {
      num_conv_layers = config["num_conv_layers"].cast<int>();
      std::cout << "Conv Layers: " << num_conv_layers << std::endl;
    }

    int num_fc_layers = 2;
    if (config.contains("num_fc_layers")) {
      num_fc_layers = config["num_fc_layers"].cast<int>();
      std::cout << "FC Layers: " << num_fc_layers << std::endl;
    }

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

    int img_d = 1;
    if (config.contains("image_depth"))
      img_d = config["image_depth"].cast<int>();

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

    size_t current_depth =
        img_d; // Start with image depth (1 for MNIST, 3 for CIFAR)

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
      }

      // CRITICAL FIX: Set kernel depth to match input depth (previous layer
      // output)
      ck.kernel_depth = current_depth;

      // Update current depth for NEXT layer
      current_depth = ck.numOfKerenels;

      arch.kernelsPerconvLayers.push_back(ck);
    }

    // D. Parse per-layer Conv activations, init functions, and distributions
    std::vector<std::string> conv_activations;
    std::vector<std::string> conv_init_functions;
    std::vector<std::string> conv_distributions;

    if (config.contains("conv_activations")) {
      try {
        conv_activations =
            config["conv_activations"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    }

    if (config.contains("conv_init_functions")) {
      try {
        conv_init_functions =
            config["conv_init_functions"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    } else if (config.contains("conv_init_types")) { // Backward compatibility
      try {
        conv_init_functions =
            config["conv_init_types"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    }

    if (config.contains("conv_distributions")) {
      try {
        conv_distributions =
            config["conv_distributions"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    }

    // Apply per-layer or use defaults
    for (int i = 0; i < num_conv_layers; ++i) {
      std::string act_str = "ReLU";
      if (i < conv_activations.size()) {
        act_str = conv_activations[i];
      }
      arch.convLayerActivationFunc.push_back(map_activation(act_str));

      std::string init_str = "Auto";
      if (i < conv_init_functions.size()) {
        init_str = conv_init_functions[i];
      }

      std::string log_msg;
      std::string resolved_init = resolve_auto_init(init_str, act_str, log_msg);
      if (!log_msg.empty()) {
        std::cout << "[Conv Layer " << i << "] " << log_msg << std::endl;
      }

      arch.convInitFunctionsType.push_back(map_init_function(resolved_init));

      std::string dist_str = "Normal";
      if (i < conv_distributions.size()) {
        dist_str = conv_distributions[i];
      }
      arch.convDistributionTypes.push_back(map_distribution(dist_str));
    }

    // Log parsed conv configuration
    if (num_conv_layers > 0) {
      std::cout << "Conv Layer Activations (enum values): ";
      for (size_t i = 0; i < arch.convLayerActivationFunc.size(); ++i) {
        std::cout << (i > 0 ? ", " : "")
                  << static_cast<int>(arch.convLayerActivationFunc[i]);
      }
      std::cout << std::endl;

      std::cout << "Conv Layer Init Types (enum values): ";
      for (size_t i = 0; i < arch.convInitFunctionsType.size(); ++i) {
        std::cout << (i > 0 ? ", " : "")
                  << static_cast<int>(arch.convInitFunctionsType[i]);
      }
      std::cout << std::endl;
    }

    // B. Pooling Layers

    // Parse pooling strides (NEW)
    std::vector<int> pooling_strides;
    if (config.contains("pooling_strides")) {
      try {
        // Handle both int and string formats
        auto py_strides = config["pooling_strides"];
        if (py::isinstance<py::list>(py_strides)) {
          for (auto item : py_strides) {
            if (py::isinstance<py::int_>(item)) {
              pooling_strides.push_back(item.cast<int>());
            } else if (py::isinstance<py::str>(item)) {
              // Parse "2x2" format - just use first number for now
              std::string str = item.cast<std::string>();
              try {
                pooling_strides.push_back(std::stoi(str));
              } catch (...) {
                pooling_strides.push_back(2); // Default
              }
            }
          }
        }
      } catch (...) {
      }
    }

    for (size_t idx = 0; idx < loop_intervals_list.size(); ++idx) {
      arch.poolingLayersInterval.push_back(loop_intervals_list[idx]);
      arch.poolingtype.push_back(pool_type);

      // Pooling kernel with configurable stride
      poolKernel pk;
      pk.filter_height = 2;
      pk.filter_width = 2;
      pk.filter_depth = 1;

      // Use configured stride or default
      if (idx < pooling_strides.size() && pooling_strides[idx] > 0) {
        pk.stride = pooling_strides[idx];
      } else {
        pk.stride = 2; // Default
      }

      arch.kernelsPerPoolingLayer.push_back(pk);
    }

    // --- END CNN Configuration Parsing ---

    // E. Parse per-layer FC activations, init functions, and distributions
    std::vector<std::string> fc_activations;
    std::vector<std::string> fc_init_functions;
    std::vector<std::string> fc_distributions;

    if (config.contains("fc_activations")) {
      try {
        fc_activations =
            config["fc_activations"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    }

    if (config.contains("fc_init_functions")) {
      try {
        fc_init_functions =
            config["fc_init_functions"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    } else if (config.contains("fc_init_types")) { // Backward compatibility
      try {
        fc_init_functions =
            config["fc_init_types"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    }

    if (config.contains("fc_distributions")) {
      try {
        fc_distributions =
            config["fc_distributions"].cast<std::vector<std::string>>();
      } catch (...) {
      }
    }

    // Apply per-layer or use defaults
    for (int i = 0; i < num_fc_layers; ++i) {
      std::string act_str = "ReLU";
      if (i < fc_activations.size()) {
        act_str = fc_activations[i];
      }
      arch.FCLayerActivationFunc.push_back(map_activation(act_str));

      std::string init_str = "Auto";
      if (i < fc_init_functions.size()) {
        init_str = fc_init_functions[i];
      }

      std::string log_msg;
      std::string resolved_init = resolve_auto_init(init_str, act_str, log_msg);
      if (!log_msg.empty()) {
        std::cout << "[FC Layer " << i << "] " << log_msg << std::endl;
      }

      arch.FCInitFunctionsType.push_back(map_init_function(resolved_init));

      std::string dist_str = "Normal";
      if (i < fc_distributions.size()) {
        dist_str = fc_distributions[i];
      }
      arch.FCDistributionTypes.push_back(map_distribution(dist_str));
    }

    // Default global distribution (deprecated but kept for safety)
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

    float beta1 = 0.9f;
    if (config.contains("beta1"))
      beta1 = config["beta1"].cast<float>();

    float beta2 = 0.999f;
    if (config.contains("beta2"))
      beta2 = config["beta2"].cast<float>();

    float beta = 0.9f;
    if (config.contains("beta"))
      beta = config["beta"].cast<float>();

    float epsilon = 1e-8f;
    if (config.contains("epsilon"))
      epsilon = config["epsilon"].cast<float>();

    arch.optConfig.learningRate = lr;
    arch.optConfig.weightDecay = decay;
    arch.optConfig.momentum = momentum;
    arch.optConfig.beta1 = beta1;
    arch.optConfig.beta2 = beta2;
    arch.optConfig.beta = beta;
    arch.optConfig.epsilon = epsilon;

    if (opt_type == "Adam")
      arch.optConfig.type = OptimizerType::opt_Adam;
    else if (opt_type == "RMSprop" || opt_type == "RMSProp")
      arch.optConfig.type = OptimizerType::opt_RMSprop;
    else if (opt_type == "SGD with Momentum" || opt_type == "SGD_Momentum")
      arch.optConfig.type = OptimizerType::opt_SGD_Momentum;
    else
      arch.optConfig.type = OptimizerType::opt_SGD;

    validate_config_cpp(num_fc_layers, neurons_list, arch, img_h, img_w,
                        num_classes);

    // Pass variable depth
    return new NNModel(arch, num_classes, img_h, img_w, img_d);

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

  // Validate dimensions against model expectations
  if (width != model.getInputWidth() || height != model.getInputHeight()) {
    std::cout << "Warning: classify_pixels received image size " << width << "x"
              << height << " but model expects " << model.getInputWidth() << "x"
              << model.getInputHeight() << std::endl;
  }

  int depth = model.getInputDepth();

  // Convert to model's image format: vector<vector<vector<double>>>
  image img_data;
  img_data.resize(depth);

  for (int d = 0; d < depth; ++d) {
    img_data[d].resize(height);
    for (int y = 0; y < height; ++y) {
      img_data[d][y].resize(width);
      for (int x = 0; x < width; ++x) {
        // Calculate source index
        // If grayscale (depth=1), stride is usually width (+pad)
        // If RGB (depth=3), pixel is usually 3-byte packed or 4-byte aligned.
        // QImage Format_RGB888 is packed 3 bytes (R,G,B).
        // QImage Format_Grayscale8 is packed 1 byte.
        // Stride is provided from Python as `bytesPerLine`.

        // Pixel start address at row y, column x
        size_t pixel_offset = y * stride + x * (depth == 1 ? 1 : 3);
        // Note: Format_RGB888 is 3 bytes per pixel. Format_RGB32 is 4.
        // Python side guarantees Format_Grayscale8 or Format_RGB888.

        // Get value for current channel d
        // For grayscale (d=0), value is ptr[offset]
        // For RGB, value is ptr[offset + d] (0=R, 1=G, 2=B)

        unsigned char val = 0;
        if ((pixel_offset + d) < info.size) {
          val = ptr[pixel_offset + d];
        }

        // Pass raw pixel values (0-255) as doubles
        img_data[d][y][x] = static_cast<double>(val);
      }
    }
  }

  return model.classify(img_data);
}

// Helper to manually destroy model (if needed, though Python usually handles
// ownership)
void destroy_model(NNModel *model) {
  if (model) {
    delete model;
  }
}

PYBIND11_MODULE(cgroot_core, m) {
  m.doc() = "CGROOT Core Neural Network Library Bindings";

  bind_definitions(m);
  bind_mnist_loader(m);
  bind_model(m);

  // Bind new factory/utility functions
  m.def("create_model", &create_model, py::return_value_policy::take_ownership,
        "Create a new NNModel from a configuration dictionary");

  m.def("classify_pixels", &classify_pixels, "Classify image from numpy array");

  m.def("destroy_model", &destroy_model, "Destroy a NNModel instance");
}
