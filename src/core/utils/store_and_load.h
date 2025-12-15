
template<typename T>
bool save4DVector(const std::vector<std::vector<std::vector<std::vector<T>>>>& vec4d,
                  const std::string& filename);



template<typename T>
bool load4DVector(std::vector<std::vector<std::vector<std::vector<T>>>>& vec4d,
                  const std::string& filename, std::streampos& bookmark);


template<typename T>
bool save2DVector(const std::vector<std::vector<T>>& vec2d,
                  const std::string& filename);



template<typename T>
bool load2DVector(std::vector<std::vector<T>>& vec2d,
                  const std::string& filename, std::streampos& bookmark);