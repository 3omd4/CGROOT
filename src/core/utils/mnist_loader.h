#pragma once

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <cstdint>

namespace cgroot {
namespace data {

/**
 * @brief MNIST dataset loader for training and testing neural networks
 * 
 * This class provides functionality to load MNIST handwritten digit dataset
 * in the original binary format. It handles both images and labels.
 */
class MNISTLoader {
public:
    /**
     * @brief Structure to hold a single MNIST image
     */
    struct MNISTImage {
        std::vector<uint8_t> pixels;  // 28x28 = 784 pixels
        uint8_t label;                // Digit label (0-9)
        
        MNISTImage() : pixels(784, 0), label(0) {}
    };

    /**
     * @brief Structure to hold the complete MNIST dataset
     */
    struct MNISTDataset {
        std::vector<MNISTImage> images;
        size_t num_images;
        size_t image_width;
        size_t image_height;
        
        MNISTDataset() : num_images(0), image_width(28), image_height(28) {}
    };

    /**
     * @brief Load training dataset from MNIST files
     * @param images_path Path to train-images-idx3-ubyte file
     * @param labels_path Path to train-labels-idx1-ubyte file
     * @return Loaded dataset or nullptr if loading failed
     */
    static std::unique_ptr<MNISTDataset> load_training_data(
        const std::string& images_path,
        const std::string& labels_path
    );

    /**
     * @brief Load test dataset from MNIST files
     * @param images_path Path to t10k-images-idx3-ubyte file
     * @param labels_path Path to t10k-labels-idx1-ubyte file
     * @return Loaded dataset or nullptr if loading failed
     */
    static std::unique_ptr<MNISTDataset> load_test_data(
        const std::string& images_path,
        const std::string& labels_path
    );

    /**
     * @brief Normalize pixel values to [0, 1] range
     * @param dataset Dataset to normalize
     */
    static void normalize_dataset(MNISTDataset& dataset);

    /**
     * @brief Convert dataset to one-hot encoded labels
     * @param dataset Dataset to convert
     * @param num_classes Number of classes (default: 10 for digits)
     */
    static void one_hot_encode_labels(MNISTDataset& dataset, size_t num_classes = 10);

    /**
     * @brief Split dataset into training and validation sets
     * @param dataset Original dataset
     * @param validation_ratio Ratio of data to use for validation (0.0-1.0)
     * @return Pair of (training_dataset, validation_dataset)
     */
    static std::pair<std::unique_ptr<MNISTDataset>, std::unique_ptr<MNISTDataset>>
    split_dataset(const MNISTDataset& dataset, float validation_ratio = 0.2f);

    /**
     * @brief Create mini-batches from dataset
     * @param dataset Source dataset
     * @param batch_size Size of each batch
     * @return Vector of batches, each containing batch_size images
     */
    static std::vector<std::vector<MNISTImage>> create_batches(
        const MNISTDataset& dataset, 
        size_t batch_size
    );

    /**
     * @brief Print dataset statistics
     * @param dataset Dataset to analyze
     */
    static void print_dataset_info(const MNISTDataset& dataset);

private:
    /**
     * @brief Internal function to load dataset from files
     * @param images_path Path to images file
     * @param labels_path Path to labels file
     * @param images_magic Expected magic number for images
     * @param labels_magic Expected magic number for labels
     * @return Loaded dataset or nullptr if loading failed
     */
    static std::unique_ptr<MNISTDataset> load_dataset(
        const std::string& images_path,
        const std::string& labels_path,
        uint32_t images_magic,
        uint32_t labels_magic
    );

    /**
     * @brief Read 32-bit integer from file in big-endian format
     * @param file Input file stream
     * @return 32-bit integer value
     */
    static uint32_t read_int32(std::ifstream& file);

    /**
     * @brief Validate MNIST file headers
     * @param file Input file stream
     * @param expected_magic Expected magic number
     * @return True if header is valid
     */
    static bool validate_header(std::ifstream& file, uint32_t expected_magic);
};

/**
 * @brief Utility functions for data preprocessing
 */
namespace utils {

/**
 * @brief Convert MNIST image to float vector (normalized)
 * @param image MNIST image
 * @return Normalized float vector
 */
std::vector<float> image_to_float_vector(const MNISTLoader::MNISTImage& image);

/**
 * @brief Convert label to one-hot vector
 * @param label Original label
 * @param num_classes Number of classes
 * @return One-hot encoded vector
 */
std::vector<float> label_to_one_hot(uint8_t label, size_t num_classes = 10);

/**
 * @brief Shuffle dataset randomly
 * @param dataset Dataset to shuffle
 * @param seed Random seed for reproducibility
 */
void shuffle_dataset(MNISTLoader::MNISTDataset& dataset, unsigned int seed = 42);

/**
 * @brief Save dataset to CSV file for inspection
 * @param dataset Dataset to save
 * @param filename Output filename
 * @param max_samples Maximum number of samples to save (0 = all)
 */
void save_to_csv(const MNISTLoader::MNISTDataset& dataset, 
                 const std::string& filename, 
                 size_t max_samples = 0);

} // namespace utils
} // namespace data
} // namespace cgroot
