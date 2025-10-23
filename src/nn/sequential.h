/*Purpose: A container to stack multiple Modules in order.

To-Do:

class Sequential : public Module.

Members: std::vector<std::shared_ptr<Module>> layers_.

sequential.h: Declare void add(std::shared_ptr<Module> module) and Tensor<T> forward(const Tensor<T>& input) override;.

sequential.cpp (add): Add module to layers_ and also add all its parameters to this Sequential's parameters_ list.

sequential.cpp (forward): Loop through layers_, passing the output of one as the input to the next.*/