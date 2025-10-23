/*Purpose: Implements the backward function.

To-Do:
Implement backward(Tensor<T>& start_tensor):
Initialize start_tensor.grad() to 1.0.
Create a std::vector<Node*> sorted_nodes.
Create a std::set<Node*> visited.
Perform a Depth First Search (DFS) starting from start_tensor.grad_fn_.get().
As the DFS recursion unwinds, add each node to sorted_nodes. This creates a topological sort.
Iterate through sorted_nodes in reverse order.
For each node, call node->_backward(node->output.grad()). (Note: You need a way to find the output tensor's gradient. This is tricky. A simpler way is to have _backward accumulate gradients into the inputs_ tensors directly.)*/