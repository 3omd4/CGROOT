#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Model Serialization Structures and Constants
// ============================================================================

#include <cstdint>

struct ModelFileHeader {
  uint32_t magic;      // 'NNMD'
  uint32_t version;    // format version
  uint32_t layerCount; // number of stored layers
};

static constexpr uint32_t MODEL_MAGIC = 0x4E4E4D44; // "NNMD"
static constexpr uint32_t MODEL_VERSION = 1;

// ============================================================================
// Model Serialization Helper Functions
// ============================================================================

// Write model file header
inline bool writeModelHeader(std::ofstream &file, uint32_t layerCount) {
  ModelFileHeader header{};
  header.magic = MODEL_MAGIC;
  header.version = MODEL_VERSION;
  header.layerCount = layerCount;

  file.write(reinterpret_cast<const char *>(&header), sizeof(header));
  return file.good();
}

// Read and validate model file header
inline bool readModelHeader(std::ifstream &file, ModelFileHeader &header) {
  file.read(reinterpret_cast<char *>(&header), sizeof(header));

  if (header.magic != MODEL_MAGIC) {
    std::cerr << "Invalid model file (bad magic number)\n";
    return false;
  }

  if (header.version != MODEL_VERSION) {
    std::cerr << "Incompatible model version (expected " << MODEL_VERSION
              << ", got " << header.version << ")\n";
    return false;
  }

  return true;
}

// Write 4D convolution kernel data (already exists as save4DVector, but
// specialized)
inline bool writeConvKernels(
    std::ofstream &file,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &kernels) {
  size_t D1 = kernels.size();
  size_t D2 = D1 ? kernels[0].size() : 0;
  size_t D3 = D2 ? kernels[0][0].size() : 0;
  size_t D4 = D3 ? kernels[0][0][0].size() : 0;

  file.write(reinterpret_cast<const char *>(&D1), sizeof(D1));
  file.write(reinterpret_cast<const char *>(&D2), sizeof(D2));
  file.write(reinterpret_cast<const char *>(&D3), sizeof(D3));
  file.write(reinterpret_cast<const char *>(&D4), sizeof(D4));

  for (size_t a = 0; a < D1; a++)
    for (size_t b = 0; b < D2; b++)
      for (size_t c = 0; c < D3; c++)
        file.write(reinterpret_cast<const char *>(kernels[a][b][c].data()),
                   D4 * sizeof(double));

  return file.good();
}

// Read 4D convolution kernel data and validate dimensions
inline bool readConvKernels(
    std::ifstream &file,
    std::vector<std::vector<std::vector<std::vector<double>>>> &kernels,
    size_t layerIndex) {
  // Read kernel dimensions
  size_t D1, D2, D3, D4;
  file.read(reinterpret_cast<char *>(&D1), sizeof(D1));
  file.read(reinterpret_cast<char *>(&D2), sizeof(D2));
  file.read(reinterpret_cast<char *>(&D3), sizeof(D3));
  file.read(reinterpret_cast<char *>(&D4), sizeof(D4));

  // Verify dimensions match
  if (kernels.size() != D1 || (D1 && kernels[0].size() != D2) ||
      (D2 && kernels[0][0].size() != D3) ||
      (D3 && kernels[0][0][0].size() != D4)) {
    std::cerr << "Kernel dimension mismatch at layer " << layerIndex << "\n";
    return false;
  }

  // Read kernel data
  for (size_t a = 0; a < D1; a++)
    for (size_t b = 0; b < D2; b++)
      for (size_t c = 0; c < D3; c++)
        file.read(reinterpret_cast<char *>(kernels[a][b][c].data()),
                  D4 * sizeof(double));

  return file.good();
}

// Write 2D neuron weights (FC or output layer)
inline bool
writeNeuronWeights(std::ofstream &file,
                   const std::vector<std::vector<double>> &neurons) {
  size_t rows = neurons.size();
  size_t cols = rows ? neurons[0].size() : 0;

  file.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  file.write(reinterpret_cast<const char *>(&cols), sizeof(cols));

  for (size_t r = 0; r < rows; r++)
    file.write(reinterpret_cast<const char *>(neurons[r].data()),
               cols * sizeof(double));

  return file.good();
}

// Read 2D neuron weights and validate dimensions
inline bool readNeuronWeights(std::ifstream &file,
                              std::vector<std::vector<double>> &neurons,
                              const std::string &layerName) {
  // Read weight dimensions
  size_t rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  // Verify dimensions match
  if (neurons.size() != rows || (rows && neurons[0].size() != cols)) {
    std::cerr << "Weight dimension mismatch at " << layerName << "\n";
    return false;
  }

  // Read weight data
  for (size_t r = 0; r < rows; r++)
    file.read(reinterpret_cast<char *>(neurons[r].data()),
              cols * sizeof(double));

  return file.good();
}
