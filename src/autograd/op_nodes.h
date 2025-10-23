/*
Purpose: Declares all concrete Node implementations.

To-Do:
class AddNode : public Node { ... }
class MulNode : public Node { ... }
class MatmulNode : public Node { ... }
class ReLUNode : public Node { ... }
class Conv2DNode : public Node { ... }
...etc. for every single operation.
Each class must override void _backward(const Tensor<T>& upstream_grad) override;.*/