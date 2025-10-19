Defines the derivative (the _backward() method) for every mathematical operation.

op_arithmetic.h: Defines AddNode, MulNode, SubNode, DivNode. Implements their _backward() methods (e.g., AddNode's backward pass just passes the gradient through).

op_linalg.h: Defines MatmulNode, TransposeNode. MatmulNode::_backward is complex, implementing the chain rule for matrix multiplication (gA = gC @ B.T and gB = A.T @ gC).

op_activation.h: Defines ReLUNode, SigmoidNode, TanhNode. Implements the derivative for each activation function (e.g., ReLUNode::_backward is grad_out * (input > 0)).

op_loss.h: Defines building-block nodes for losses, like SoftmaxNode, LogNode, MeanNode.

op_nn.h: Defines Conv2DNode, PoolNode. Conv2DNode::_backward will be one of the most complex files, implementing the derivatives with respect to the input, weights, and bias.