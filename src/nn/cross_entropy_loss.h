/*Purpose: Cross Entropy loss function.

To-Do:

class CrossEntropyLoss : public Module.

cross_entropy_loss.h: Declare Tensor<T> forward(const Tensor<T>& logits, const Tensor<T>& targets).

cross_entropy_loss.cpp (forward): This is the most complex loss. Implement the "LogSoftmax" trick: log_softmax = logits - logits.exp().sum().log(). Then use the targets tensor to "gather" the correct log probabilities. Return the negative mean.*/