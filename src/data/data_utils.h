#pragma once

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>

namespace cgroot {
namespace data {

/**
 * @brief General-purpose data loading and preprocessing utilities
 */
class DataUtils {
public:
    /**
     * @brief Load data from CSV file
     * @param filename Path to CSV file
     * @param has_header Whether the CSV has a header row
     * @param delimiter Character used as delimiter
     * @return Vector of rows, each row is a vector of strings
     */
    static std::vector<std::vector<std::string>> load_csv(
        const std::string& filename,
        bool has_header = true,
        char delimiter = ','
    );

    /**
     * @brief Save data to CSV file
     * @param data Vector of rows to save
     * @param filename Output filename
     * @param header Optional header row
     * @param delimiter Character to use as delimiter
     */
    static void save_csv(
        const std::vector<std::vector<std::string>>& data,
        const std::string& filename,
        const std::vector<std::string>& header = {},
        char delimiter = ','
    );

    /**
     * @brief Split dataset into training and testing sets
     * @param data Input dataset
     * @param test_ratio Ratio of data to use for testing (0.0-1.0)
     * @param random_seed Seed for random number generator
     * @return Pair of (training_data, testing_data)
     */
    template<typename T>
    static std::pair<std::vector<T>, std::vector<T>> train_test_split(
        const std::vector<T>& data,
        float test_ratio = 0.2f,
        unsigned int random_seed = 42
    );

    /**
     * @brief Normalize data to [0, 1] range
     * @param data Input data
     * @return Normalized data
     */
    static std::vector<std::vector<float>> normalize_data(
        const std::vector<std::vector<float>>& data
    );

    /**
     * @brief Standardize data (zero mean, unit variance)
     * @param data Input data
     * @return Standardized data
     */
    static std::vector<std::vector<float>> standardize_data(
        const std::vector<std::vector<float>>& data
    );

    /**
     * @brief Create mini-batches from dataset
     * @param data Input dataset
     * @param batch_size Size of each batch
     * @param shuffle Whether to shuffle data before batching
     * @param random_seed Seed for shuffling
     * @return Vector of batches
     */
    template<typename T>
    static std::vector<std::vector<T>> create_batches(
        const std::vector<T>& data,
        size_t batch_size,
        bool shuffle = true,
        unsigned int random_seed = 42
    );

    /**
     * @brief Shuffle dataset randomly
     * @param data Dataset to shuffle
     * @param random_seed Seed for random number generator
     */
    template<typename T>
    static void shuffle_data(std::vector<T>& data, unsigned int random_seed = 42);

    /**
     * @brief Convert string vector to float vector
     * @param str_vec String vector
     * @return Float vector
     */
    static std::vector<float> string_to_float_vector(const std::vector<std::string>& str_vec);

    /**
     * @brief Convert float vector to string vector
     * @param float_vec Float vector
     * @param precision Decimal precision
     * @return String vector
     */
    static std::vector<std::string> float_to_string_vector(
        const std::vector<float>& float_vec,
        int precision = 6
    );

    /**
     * @brief Check if file exists
     * @param filename File path
     * @return True if file exists
     */
    static bool file_exists(const std::string& filename);

    /**
     * @brief Get file size in bytes
     * @param filename File path
     * @return File size in bytes, -1 if error
     */
    static long long get_file_size(const std::string& filename);

    /**
     * @brief Create directory if it doesn't exist
     * @param dir_path Directory path
     * @return True if directory exists or was created successfully
     */
    static bool create_directory(const std::string& dir_path);

    /**
     * @brief List files in directory
     * @param dir_path Directory path
     * @param extension Optional file extension filter
     * @return Vector of filenames
     */
    static std::vector<std::string> list_files(
        const std::string& dir_path,
        const std::string& extension = ""
    );

private:
    /**
     * @brief Split string by delimiter
     * @param str Input string
     * @param delimiter Delimiter character
     * @return Vector of tokens
     */
    static std::vector<std::string> split_string(
        const std::string& str,
        char delimiter
    );

    /**
     * @brief Trim whitespace from string
     * @param str Input string
     * @return Trimmed string
     */
    static std::string trim_string(const std::string& str);
};

/**
 * @brief Data augmentation utilities for image data
 */
namespace augmentation {

/**
 * @brief Rotate image by specified angle
 * @param image Input image (flattened)
 * @param width Image width
 * @param height Image height
 * @param angle Rotation angle in degrees
 * @return Rotated image
 */
std::vector<float> rotate_image(
    const std::vector<float>& image,
    int width,
    int height,
    float angle
);

/**
 * @brief Flip image horizontally
 * @param image Input image (flattened)
 * @param width Image width
 * @param height Image height
 * @return Horizontally flipped image
 */
std::vector<float> flip_horizontal(
    const std::vector<float>& image,
    int width,
    int height
);

/**
 * @brief Add random noise to image
 * @param image Input image
 * @param noise_level Noise level (0.0-1.0)
 * @param random_seed Seed for random number generator
 * @return Noisy image
 */
std::vector<float> add_noise(
    const std::vector<float>& image,
    float noise_level = 0.1f,
    unsigned int random_seed = 42
);

/**
 * @brief Apply random brightness adjustment
 * @param image Input image
 * @param brightness_factor Brightness multiplier
 * @param random_seed Seed for random number generator
 * @return Brightness adjusted image
 */
std::vector<float> adjust_brightness(
    const std::vector<float>& image,
    float brightness_factor = 1.0f,
    unsigned int random_seed = 42
);

} // namespace augmentation

} // namespace data
} // namespace cgroot
