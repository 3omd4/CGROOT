#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


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
      .def_readwrite("learningRate", &architecture::learningRate)
      .def_readwrite("batch_size", &architecture::batch_size)
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
      .def("train", &NNModel::train)
      .def("train_batch", &NNModel::train_batch)
      .def("classify", &NNModel::classify)
      .def("getProbabilities", &NNModel::getProbabilities)
      .def("train_epochs", &NNModel::train_epochs, py::arg("dataset"),
           py::arg("config"), py::arg("progress_callback") = py::none(),
           py::arg("log_callback") = py::none(),
           "Train model for multiple epochs with callbacks");
}

PYBIND11_MODULE(cgroot_core, m) {
  m.doc() = "CGROOT Core Neural Network Library Bindings";

  bind_definitions(m);
  bind_mnist_loader(m);
  bind_model(m);
}
