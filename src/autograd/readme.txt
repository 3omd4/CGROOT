Holds the entire automatic differentiation engine. This is the "brain" that computes gradients.

node.h: Defines the abstract base class Node. This is the interface for every operation in the computational graph (e.g., AddNode, MatmulNode). It requires every child class to implement a _backward() method.

graph.cpp: Implements the Tensor::backward() logic. This file contains the topological sort algorithm. When loss.backward() is called, this code traverses the graph of Node objects from end to beginning and calls _backward() on each one in the correct order.

ops/: A sub-folder containing the concrete implementations of each Node.