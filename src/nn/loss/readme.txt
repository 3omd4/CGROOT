Defines user-friendly loss functions.

mse_loss.h: Defines nn::MSELoss. Its forward(y_pred, y_true) method computes (y_pred - y_true).pow(2).mean().

cross_entropy_loss.h: Defines nn::CrossEntropyLoss. This combines LogSoftmax and NLLLoss into a single, stable operation.