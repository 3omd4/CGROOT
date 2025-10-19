Contains the standard neural network layers that have learnable weights.

linear.h / .cpp: Defines nn::Linear. Inherits from nn::Module. Its constructor creates weight and bias Parameters. Its forward() method calls input.matmul(weight.transpose()) + bias.

conv2d.h / .cpp: Defines nn::Conv2D. Inherits from nn::Module. Its constructor creates weight and bias. Its forward() method calls input.conv2d(...).

sequential.h: Defines the nn::Sequential container, a Module that holds a list of other Modules and calls them in order.