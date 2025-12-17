#include "mnist_loader.h"
#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

namespace cgroot {
namespace data {

std::unique_ptr<MNISTLoader::MNISTDataset>
MNISTLoader::load_training_data(const std::string &images_path,
                                const std::string &labels_path) {
  return load_dataset(images_path, labels_path, 2051,
                      2049); // MNIST magic numbers
}

std::unique_ptr<MNISTLoader::MNISTDataset>
MNISTLoader::load_test_data(const std::string &images_path,
                            const std::string &labels_path) {
  return load_dataset(images_path, labels_path, 2051,
                      2049); // Same magic numbers
}

std::unique_ptr<MNISTLoader::MNISTDataset>
MNISTLoader::load_dataset(const std::string &images_path,
                          const std::string &labels_path, uint32_t images_magic,
                          uint32_t labels_magic) {
  auto dataset = std::make_unique<MNISTDataset>();

  // Open images file
  std::ifstream images_file(images_path, std::ios::binary);
  if (!images_file.is_open()) {
    std::cerr << "Error: Could not open images file: " << images_path
              << std::endl;
    return nullptr;
  }

  // Open labels file
  std::ifstream labels_file(labels_path, std::ios::binary);
  if (!labels_file.is_open()) {
    std::cerr << "Error: Could not open labels file: " << labels_path
              << std::endl;
    return nullptr;
  }

  // Read images metadata
  uint32_t magic = read_int32(images_file);

  // Validate magic number (2051 for 3D, 2052 for 4D/Color)
  if (magic != images_magic && magic != 2052) {
    std::cerr << "Error: Invalid images file header. Magic: " << magic
              << ". Expected: " << images_magic << " or 2052" << std::endl;
    return nullptr;
  }

  uint32_t num_images = read_int32(images_file);

  uint32_t rows = 0;
  uint32_t cols = 0;
  uint32_t depth = 1;

  if (magic == 2052) {
    // IDX4: Num, Depth, Rows, Cols
    depth = read_int32(images_file);
    rows = read_int32(images_file);
    cols = read_int32(images_file);
  } else {
    // IDX3: Num, Rows, Cols
    rows = read_int32(images_file);
    cols = read_int32(images_file);
  }

  // Read and validate labels header
  if (!validate_header(labels_file, labels_magic)) {
    std::cerr << "Error: Invalid labels file header" << std::endl;
    return nullptr;
  }

  // Read labels metadata
  uint32_t num_labels = read_int32(labels_file);

  // Validate data consistency
  if (num_images != num_labels) {
    std::cerr << "Error: Number of images (" << num_images
              << ") doesn't match number of labels (" << num_labels << ")"
              << std::endl;
    return nullptr;
  }

  // We support arbitrary sizes now.
  // if (rows != 28 || cols != 28) { ... }

  // Set dataset properties
  dataset->num_images = num_images;
  dataset->image_width = cols;
  dataset->image_height = rows;
  dataset->depth = depth;
  dataset->images.reserve(num_images);

  // Read images and labels
  // Read images and labels
  size_t image_size = rows * cols * depth;
  for (uint32_t i = 0; i < num_images; ++i) {
    MNISTImage image;
    image.pixels.resize(image_size);

    // Read image pixels
    images_file.read(reinterpret_cast<char *>(image.pixels.data()), image_size);
    if (images_file.gcount() != static_cast<std::streamsize>(image_size)) {
      std::cerr << "Error: Failed to read image " << i << std::endl;
      return nullptr;
    }

    // Read label
    labels_file.read(reinterpret_cast<char *>(&image.label), 1);
    if (labels_file.gcount() != 1) {
      std::cerr << "Error: Failed to read label " << i << std::endl;
      return nullptr;
    }

    dataset->images.push_back(std::move(image));
  }

  std::cout << "Successfully loaded " << num_images << " MNIST samples"
            << std::endl;
  return dataset;
}

void MNISTLoader::one_hot_encode_labels(MNISTDataset &dataset,
                                        size_t num_classes) {
  // This would be implemented when we have the tensor class
  // For now, we'll just store the original labels
  std::cout << "One-hot encoding not yet implemented (requires tensor class)"
            << std::endl;
}

std::pair<std::unique_ptr<MNISTLoader::MNISTDataset>,
          std::unique_ptr<MNISTLoader::MNISTDataset>>
MNISTLoader::split_dataset(const MNISTDataset &dataset,
                           float validation_ratio) {
  auto train_dataset = std::make_unique<MNISTDataset>();
  auto val_dataset = std::make_unique<MNISTDataset>();

  size_t val_size = static_cast<size_t>(dataset.num_images * validation_ratio);
  size_t train_size = dataset.num_images - val_size;

  train_dataset->num_images = train_size;
  train_dataset->image_width = dataset.image_width;
  train_dataset->image_height = dataset.image_height;
  train_dataset->images.reserve(train_size);

  val_dataset->num_images = val_size;
  val_dataset->image_width = dataset.image_width;
  val_dataset->image_height = dataset.image_height;
  val_dataset->images.reserve(val_size);

  // Split the data
  for (size_t i = 0; i < train_size; ++i) {
    train_dataset->images.push_back(dataset.images[i]);
  }

  for (size_t i = train_size; i < dataset.num_images; ++i) {
    val_dataset->images.push_back(dataset.images[i]);
  }

  return std::make_pair(std::move(train_dataset), std::move(val_dataset));
}

std::vector<std::vector<MNISTLoader::MNISTImage>>
MNISTLoader::create_batches(const MNISTDataset &dataset, size_t batch_size) {
  std::vector<std::vector<MNISTImage>> batches;

  for (size_t i = 0; i < dataset.num_images; i += batch_size) {
    std::vector<MNISTImage> batch;
    size_t end_idx = std::min(i + batch_size, dataset.num_images);

    for (size_t j = i; j < end_idx; ++j) {
      batch.push_back(dataset.images[j]);
    }

    batches.push_back(std::move(batch));
  }

  return batches;
}

void MNISTLoader::print_dataset_info(const MNISTDataset &dataset) {
  std::cout << "Dataset Information:" << std::endl;
  std::cout << "  Number of images: " << dataset.num_images << std::endl;
  std::cout << "  Image dimensions: " << dataset.image_width << "x"
            << dataset.image_height << std::endl;

  // Count label distribution
  std::vector<size_t> label_counts(10, 0);
  for (const auto &image : dataset.images) {
    if (image.label < 10) {
      label_counts[image.label]++;
    }
  }

  std::cout << "  Label distribution:" << std::endl;
  for (size_t i = 0; i < 10; ++i) {
    std::cout << "    " << i << ": " << label_counts[i] << " samples"
              << std::endl;
  }
}

uint32_t MNISTLoader::read_int32(std::ifstream &file) {
  uint32_t value;
  file.read(reinterpret_cast<char *>(&value), sizeof(value));

  // Convert from big-endian to little-endian
  return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
         ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
}

bool MNISTLoader::validate_header(std::ifstream &file,
                                  uint32_t expected_magic) {
  uint32_t magic = read_int32(file);
  return magic == expected_magic;
}

// Utility functions implementation
namespace utils {

std::vector<float> image_to_float_vector(const MNISTLoader::MNISTImage &image) {
  std::vector<float> result;
  result.reserve(image.pixels.size());

  for (uint8_t pixel : image.pixels) {
    result.push_back(static_cast<float>(pixel) / 255.0f);
  }

  return result;
}

std::vector<float> label_to_one_hot(uint8_t label, size_t num_classes) {
  std::vector<float> one_hot(num_classes, 0.0f);
  if (label < num_classes) {
    one_hot[label] = 1.0f;
  }
  return one_hot;
}

void shuffle_dataset(MNISTLoader::MNISTDataset &dataset, unsigned int seed) {
  std::mt19937 rng(seed);
  std::shuffle(dataset.images.begin(), dataset.images.end(), rng);
}

void save_to_csv(const MNISTLoader::MNISTDataset &dataset,
                 const std::string &filename, size_t max_samples) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filename
              << std::endl;
    return;
  }

  size_t samples_to_save = (max_samples == 0)
                               ? dataset.num_images
                               : std::min(max_samples, dataset.num_images);

  // Write header
  file << "label";
  for (size_t i = 0; i < dataset.image_width * dataset.image_height; ++i) {
    file << ",pixel" << i;
  }
  file << std::endl;

  // Write data
  for (size_t i = 0; i < samples_to_save; ++i) {
    const auto &image = dataset.images[i];
    file << static_cast<int>(image.label);

    for (auto pixel : image.pixels) {
      file << "," << static_cast<int>(pixel);
    }
    file << std::endl;
  }

  std::cout << "Saved " << samples_to_save << " samples to " << filename
            << std::endl;
}

} // namespace utils
} // namespace data
} // namespace cgroot