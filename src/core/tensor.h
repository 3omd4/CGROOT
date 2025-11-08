#ifndef CGROOT_SRC_CORE_TENSOR_H_
#define CGROOT_SRC_CORE_TENSOR_H_
#include <vector>
#include <cstddef>

template<typename T>
class Tensor {
private:
    std::vector<T> data_;
    std::vector<size_t> shape_;
    std::vector<T> grad_;

public:
    Tensor();
    Tensor(const std::vector<size_t>& shape, T init_val = static_cast<T>(0));

    const std::vector<size_t>& shape() const;
    std::vector<T>& data();
    const std::vector<T>& data() const;

    Tensor<T>& grad();
    void zero_grad();

    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator-(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator*(const T& scalar) const;
    Tensor<T> operator/(const T& scalar) const;
    Tensor<T> operator+(T scalar) const;
    Tensor<T> sqrt() const;

    Tensor<T>& operator=(const Tensor<T>& other);

private:
    void check_shape_match(const Tensor<T>& other) const;
};

#endif
/*
Purpose: The primary data structure of the entire library.
To-Do:
Define a template <typename T> class Tensor.

Private Members:
std::shared_ptr<T[]> data_: Smart pointer to the raw data buffer on the CPU.
Shape shape_: The shape object.
std::vector<size_t> strides_: Strides for this tensor (enables 0-copy views).
bool requires_grad_ = false: Flag for autograd.
Tensor<T> grad_: A tensor to store the accumulated gradient.
std::shared_ptr<autograd::Node> grad_fn_: Pointer to the autograd node that created this tensor.

Public Methods (API):
Constructors: Tensor(Shape shape, bool requires_grad = false), Tensor::create_from_data(...).
Accessors: shape(), strides(), data(), grad(), requires_grad().
Autograd: backward().
Math Ops: add(const Tensor&), sub(...), mul(...), div(...), matmul(const Tensor&), conv2d(...).
Activation Ops: relu(), sigmoid(), tanh().
Loss/Reduction Ops: pow(float exp), mean(), sum(), log(), exp().
Manipulation Ops: transpose(int dim0, int dim1).
Helpers: zero_grad() (zeros the grad_ tensor).
*/