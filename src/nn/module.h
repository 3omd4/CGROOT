/*Purpose: Defines the Module base class (like torch.nn.Module).

To-Do:
class Module.
Members: std::vector<Parameter<T>*> parameters_.
Methods:
virtual ~Module() {}
virtual Tensor<T> forward(const Tensor<T>& input) = 0; (Or other signatures).
std::vector<Parameter<T>*> parameters(): Returns parameters_.
void zero_grad(): Loops through parameters_ and calls p->zero_grad().
protected: void register_parameter(Parameter<T>* param): Helper to add a parameter to the list.*/