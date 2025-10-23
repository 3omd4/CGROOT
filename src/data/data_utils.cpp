#include "data_utils.h"
#include <sstream>
#include <filesystem>
#include <random>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cgroot {
namespace data {

std::vector<std::vector<std::string>> DataUtils::load_csv(
    const std::string& filename,
    bool has_header,
    char delimiter
) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return data;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line && has_header) {
            first_line = false;
            continue; // Skip header
        }
        
        std::vector<std::string> row = split_string(line, delimiter);
        for (auto& cell : row) {
            cell = trim_string(cell);
        }
        data.push_back(row);
    }
    
    return data;
}

void DataUtils::save_csv(
    const std::vector<std::vector<std::string>>& data,
    const std::string& filename,
    const std::vector<std::string>& header,
    char delimiter
) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not create file: " << filename << std::endl;
        return;
    }
    
    // Write header if provided
    if (!header.empty()) {
        for (size_t i = 0; i < header.size(); ++i) {
            if (i > 0) file << delimiter;
            file << header[i];
        }
        file << std::endl;
    }
    
    // Write data
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) file << delimiter;
            file << row[i];
        }
        file << std::endl;
    }
}

template<typename T>
std::pair<std::vector<T>, std::vector<T>> DataUtils::train_test_split(
    const std::vector<T>& data,
    float test_ratio,
    unsigned int random_seed
) {
    std::vector<T> shuffled_data = data;
    shuffle_data(shuffled_data, random_seed);
    
    size_t test_size = static_cast<size_t>(data.size() * test_ratio);
    size_t train_size = data.size() - test_size;
    
    std::vector<T> train_data(shuffled_data.begin(), shuffled_data.begin() + train_size);
    std::vector<T> test_data(shuffled_data.begin() + train_size, shuffled_data.end());
    
    return std::make_pair(train_data, test_data);
}

std::vector<std::vector<float>> DataUtils::normalize_data(
    const std::vector<std::vector<float>>& data
) {
    if (data.empty()) return data;
    
    std::vector<std::vector<float>> normalized_data = data;
    size_t num_features = data[0].size();
    
    for (size_t j = 0; j < num_features; ++j) {
        // Find min and max for this feature
        float min_val = data[0][j];
        float max_val = data[0][j];
        
        for (const auto& row : data) {
            min_val = std::min(min_val, row[j]);
            max_val = std::max(max_val, row[j]);
        }
        
        // Normalize to [0, 1]
        float range = max_val - min_val;
        if (range > 0) {
            for (auto& row : normalized_data) {
                row[j] = (row[j] - min_val) / range;
            }
        }
    }
    
    return normalized_data;
}

std::vector<std::vector<float>> DataUtils::standardize_data(
    const std::vector<std::vector<float>>& data
) {
    if (data.empty()) return data;
    
    std::vector<std::vector<float>> standardized_data = data;
    size_t num_features = data[0].size();
    
    for (size_t j = 0; j < num_features; ++j) {
        // Calculate mean
        float sum = 0.0f;
        for (const auto& row : data) {
            sum += row[j];
        }
        float mean = sum / data.size();
        
        // Calculate standard deviation
        float sum_sq_diff = 0.0f;
        for (const auto& row : data) {
            float diff = row[j] - mean;
            sum_sq_diff += diff * diff;
        }
        float std_dev = std::sqrt(sum_sq_diff / data.size());
        
        // Standardize
        if (std_dev > 0) {
            for (auto& row : standardized_data) {
                row[j] = (row[j] - mean) / std_dev;
            }
        }
    }
    
    return standardized_data;
}

template<typename T>
std::vector<std::vector<T>> DataUtils::create_batches(
    const std::vector<T>& data,
    size_t batch_size,
    bool shuffle,
    unsigned int random_seed
) {
    std::vector<T> data_copy = data;
    
    if (shuffle) {
        shuffle_data(data_copy, random_seed);
    }
    
    std::vector<std::vector<T>> batches;
    
    for (size_t i = 0; i < data_copy.size(); i += batch_size) {
        std::vector<T> batch;
        size_t end_idx = std::min(i + batch_size, data_copy.size());
        
        for (size_t j = i; j < end_idx; ++j) {
            batch.push_back(data_copy[j]);
        }
        
        batches.push_back(std::move(batch));
    }
    
    return batches;
}

template<typename T>
void DataUtils::shuffle_data(std::vector<T>& data, unsigned int random_seed) {
    std::mt19937 rng(random_seed);
    std::shuffle(data.begin(), data.end(), rng);
}

std::vector<float> DataUtils::string_to_float_vector(const std::vector<std::string>& str_vec) {
    std::vector<float> float_vec;
    float_vec.reserve(str_vec.size());
    
    for (const auto& str : str_vec) {
        try {
            float_vec.push_back(std::stof(str));
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not convert '" << str << "' to float: " << e.what() << std::endl;
            float_vec.push_back(0.0f);
        }
    }
    
    return float_vec;
}

std::vector<std::string> DataUtils::float_to_string_vector(
    const std::vector<float>& float_vec,
    int precision
) {
    std::vector<std::string> str_vec;
    str_vec.reserve(float_vec.size());
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);
    
    for (float val : float_vec) {
        oss.str("");
        oss << val;
        str_vec.push_back(oss.str());
    }
    
    return str_vec;
}

bool DataUtils::file_exists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

long long DataUtils::get_file_size(const std::string& filename) {
    try {
        return std::filesystem::file_size(filename);
    } catch (const std::exception&) {
        return -1;
    }
}

bool DataUtils::create_directory(const std::string& dir_path) {
    try {
        return std::filesystem::create_directories(dir_path);
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> DataUtils::list_files(
    const std::string& dir_path,
    const std::string& extension
) {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (extension.empty() || filename.substr(filename.find_last_of(".") + 1) == extension) {
                    files.push_back(filename);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing files in " << dir_path << ": " << e.what() << std::endl;
    }
    
    return files;
}

std::vector<std::string> DataUtils::split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::string DataUtils::trim_string(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// Data augmentation implementations
namespace augmentation {

std::vector<float> rotate_image(
    const std::vector<float>& image,
    int width,
    int height,
    float angle
) {
    std::vector<float> rotated(width * height, 0.0f);
    
    float radians = angle * M_PI / 180.0f;
    float cos_angle = std::cos(radians);
    float sin_angle = std::sin(radians);
    
    int center_x = width / 2;
    int center_y = height / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Translate to origin
            int dx = x - center_x;
            int dy = y - center_y;
            
            // Apply rotation
            int new_x = static_cast<int>(dx * cos_angle - dy * sin_angle) + center_x;
            int new_y = static_cast<int>(dx * sin_angle + dy * cos_angle) + center_y;
            
            // Check bounds and copy pixel
            if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                rotated[y * width + x] = image[new_y * width + new_x];
            }
        }
    }
    
    return rotated;
}

std::vector<float> flip_horizontal(
    const std::vector<float>& image,
    int width,
    int height
) {
    std::vector<float> flipped = image;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width / 2; ++x) {
            int left_idx = y * width + x;
            int right_idx = y * width + (width - 1 - x);
            std::swap(flipped[left_idx], flipped[right_idx]);
        }
    }
    
    return flipped;
}

std::vector<float> add_noise(
    const std::vector<float>& image,
    float noise_level,
    unsigned int random_seed
) {
    std::vector<float> noisy = image;
    std::mt19937 rng(random_seed);
    std::normal_distribution<float> noise_dist(0.0f, noise_level);
    
    for (auto& pixel : noisy) {
        pixel += noise_dist(rng);
        pixel = std::max(0.0f, std::min(1.0f, pixel)); // Clamp to [0, 1]
    }
    
    return noisy;
}

std::vector<float> adjust_brightness(
    const std::vector<float>& image,
    float brightness_factor,
    unsigned int random_seed
) {
    std::vector<float> adjusted = image;
    
    for (auto& pixel : adjusted) {
        pixel *= brightness_factor;
        pixel = std::max(0.0f, std::min(1.0f, pixel)); // Clamp to [0, 1]
    }
    
    return adjusted;
}

} // namespace augmentation
} // namespace data
} // namespace cgroot
