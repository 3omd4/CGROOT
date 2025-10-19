Holds the most fundamental components. This folder defines what a Tensor is and the raw math to operate on it. It knows nothing about autograd.

tensor.h / tensor.cpp: The most important class.

tensor.h: Defines the Tensor class interface. This class holds the data pointer (e.g., std::vector<T>), the shape, and the strides. It also defines the public-facing methods users will call (e.g., matmul(), add(), conv2d(), backward()).

tensor.cpp: Implements the Tensor methods. When a user calls a.matmul(b), this file's code will:

Call the cpu_matmul kernel from src/core/math/.

Create a MatmulNode (from src/autograd/ops/) to link the new Tensor into the computational graph.

shape.h: A crucial helper class/file. It contains all the logic for calculating strides, broadcasting shapes, and computing output shapes for operations like conv2d and pool.

math/: This sub-folder contains the "from-scratch" math implementations.