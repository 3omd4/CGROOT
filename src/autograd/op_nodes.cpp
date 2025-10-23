/*Purpose: Implements the _backward method for every node.

To-Do:
AddNode::_backward: Accumulate gradient: inputs_[0]->grad() += upstream_grad; and inputs_[1]->grad() += upstream_grad; (Handle broadcasting).
MulNode::_backward: Chain rule: inputs_[0]->grad() += inputs_[1]->data() * upstream_grad; and inputs_[1]->grad() += inputs_[0]->data() * upstream_grad;.
MatmulNode::_backward: Matrix chain rule. inputs_[0]->grad() += upstream_grad.matmul(inputs_[1]->transpose()); and inputs_[1]->grad() += inputs_[0]->transpose().matmul(upstream_grad);.
ReLUNode::_backward: Call the kernel: math::cpu_relu_backward(inputs_[0]->grad(), upstream_grad, inputs_[0]->data(), ...);.*/