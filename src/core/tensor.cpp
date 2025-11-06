#include "tensor.h"
#include <cmath>
#include <stdexcept>

template<typename T>
Tensor<T>::Tensor() {}

template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, T init_val)
    : shape_(shape) {
    size_t total = 1;
    for (size_t i = 0; i < shape.size(); ++i)
        total *= shape[i];
    data_.assign(total, init_val);
    grad_.assign(total, static_cast<T>(0));
}

template<typename T>
const std::vector<size_t>& Tensor<T>::shape() const { return shape_; }

template<typename T>
std::vector<T>& Tensor<T>::data() { return data_; }

template<typename T>
const std::vector<T>& Tensor<T>::data() const { return data_; }

template<typename T>
Tensor<T>& Tensor<T>::grad() {
    static Tensor<T> grad_ref;
    grad_ref.data_ = grad_;
    grad_ref.shape_ = shape_;
    return grad_ref;
}

template<typename T>
void Tensor<T>::zero_grad() {
    for (size_t i = 0; i < grad_.size(); ++i)
        grad_[i] = static_cast<T>(0);
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    check_shape_match(other);
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = data_[i] + other.data_[i];
    return out;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
    check_shape_match(other);
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = data_[i] - other.data_[i];
    return out;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    check_shape_match(other);
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = data_[i] * other.data_[i];
    return out;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const T& scalar) const {
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = data_[i] * scalar;
    return out;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const T& scalar) const {
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = data_[i] / scalar;
    return out;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T scalar) const {
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = data_[i] + scalar;
    return out;
}

template<typename T>
Tensor<T> Tensor<T>::sqrt() const {
    Tensor<T> out(shape_);
    for (size_t i = 0; i < data_.size(); ++i)
        out.data_[i] = std::sqrt(data_[i]);
    return out;
}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other) {
    if (this == &other) return *this;
    shape_ = other.shape_;
    data_ = other.data_;
    grad_ = other.grad_;
    return *this;
}

template<typename T>
void Tensor<T>::check_shape_match(const Tensor<T>& other) const {
    if (shape_ != other.shape_)
        throw std::runtime_error("Tensor shape mismatch.");
}

// Explicit instantiations
template class Tensor<float>;
template class Tensor<double>;

/*
Purpose: Implementation of all Tensor methods.

To-Do:
Implement all constructors. They must allocate memory for data_ and compute strides_.
Implement backward(). This method will call autograd::backward(*this).
Implement every Math Op (e.g., add):
Create a new Tensor (result) for the output.
Call the corresponding kernel (e.g., math::cpu_add(result.data(), this->data(), other.data(), ...)).
If requires_grad_ is true for either input:
Create an autograd node: result.grad_fn_ = std::make_shared<autograd::AddNode>(*this, other).
Set result.requires_grad_ = true.
Return result.
*/