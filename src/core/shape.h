/*
Here is a complete, file-by-file blueprint for the CGroot++ project. The descriptions are designed to be actionable for an engineer to pick up and begin implementation.
src/core/
This folder contains the fundamental data structures: the Tensor and its properties.
src/core/shape.h
Purpose: A helper class to manage tensor shapes, strides, and broadcasting.

To-Do:
Define a class Shape (or struct Shape).
Members: std::vector<size_t> dims_.

Methods:
Shape(std::vector<size_t> dims): Constructor.
size_t ndim() const: Returns number of dimensions.
size_t total_size() const: Returns total number of elements.
std::vector<size_t> compute_strides() const: Calculates and returns the strides vector for a contiguous tensor.
static Shape broadcast_shapes(const Shape& a, const Shape& b): (Advanced) Static helper to compute the broadcasted shape.
src/core/tensor.h
*/