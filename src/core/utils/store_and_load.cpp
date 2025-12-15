#include <vector>
#include <fstream>
#include <iostream>

#include "store_and_load.h"

template<typename T>
bool save4DVector(const std::vector<std::vector<std::vector<std::vector<T>>>>& vec4d,
                  const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return false;
    }

    size_t D1 = vec4d.size();
    size_t D2 = D1 ? vec4d[0].size() : 0;
    size_t D3 = (D2 ? vec4d[0][0].size() : 0);
    size_t D4 = (D3 ? vec4d[0][0][0].size() : 0);

    // Write dimensions
    file.write((char*)&D1, sizeof(D1));
    file.write((char*)&D2, sizeof(D2));
    file.write((char*)&D3, sizeof(D3));
    file.write((char*)&D4, sizeof(D4));

    // Write raw data
    for (size_t i = 0; i < D1; i++)
        for (size_t j = 0; j < D2; j++)
            for (size_t k = 0; k < D3; k++)
                file.write((char*)vec4d[i][j][k].data(), D4 * sizeof(T));

    return true;
}

template<typename T>
bool load4DVector(std::vector<std::vector<std::vector<std::vector<T>>>>& vec4d,
                  const std::string& filename, std::streampos& bookmark)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading: " << filename << "\n";
        return false;
    }

    if(bookmark == std::streampos(-1))
    {
        return false;
    }

    file.seekg(bookmark);

    size_t D1, D2, D3, D4;

    // Read dimensions
    file.read((char*)&D1, sizeof(D1));
    file.read((char*)&D2, sizeof(D2));
    file.read((char*)&D3, sizeof(D3));
    file.read((char*)&D4, sizeof(D4));

    vec4d.assign(D1,
        std::vector<std::vector<std::vector<T>>>(D2,
            std::vector<std::vector<T>>(D3,
                std::vector<T>(D4))));

    // Read raw data
    for (size_t i = 0; i < D1; i++)
        for (size_t j = 0; j < D2; j++)
            for (size_t k = 0; k < D3; k++)
                file.read((char*)vec4d[i][j][k].data(), D4 * sizeof(T));

    
    file.clear();

    bookmark = file.tellg();

    return true;
}

template<typename T>
bool save2DVector(const std::vector<std::vector<T>>& vec2d,
                  const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << "\n";
        return false;
    }

    size_t rows = vec2d.size();
    size_t cols = rows ? vec2d[0].size() : 0;

    // Write dimensions
    file.write((char*)&rows, sizeof(rows));
    file.write((char*)&cols, sizeof(cols));

    // Write each row as raw binary
    for (size_t r = 0; r < rows; r++)
        file.write((char*)vec2d[r].data(), cols * sizeof(T));

    return true;
}


template<typename T>
bool load2DVector(std::vector<std::vector<T>>& vec2d,
                  const std::string& filename, std::streampos& bookmark)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file for reading: " << filename << "\n";
        return false;
    }

    if(bookmark == std::streampos(-1))
    {
        return false;
    }

    file.seekg(bookmark);



    size_t rows, cols;

    // Read dimensions
    file.read((char*)&rows, sizeof(rows));
    file.read((char*)&cols, sizeof(cols));

    // Resize target vector
    vec2d.assign(rows, std::vector<T>(cols));

    // Read row-by-row
    for (size_t r = 0; r < rows; r++)
        file.read((char*)vec2d[r].data(), cols * sizeof(T));
    
    file.clear();

    bookmark = file.tellg();

    return true;
}
