Contains the optimization algorithms that update model weights.

optimizer.h: Defines the abstract base class optim::Optimizer. Its constructor takes the list of model parameters from model.parameters().

sgd.h / .cpp: Defines optim::SGD. Implements the step() method to perform the weight = weight - learning_rate * weight.grad update.

adam.h / .cpp: Defines optim::Adam. Implements the step() method using momentum and squared-gradient-based updates.