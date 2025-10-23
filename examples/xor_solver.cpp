/*
Purpose: Prove the framework can train an MLP.

To-Do:

Include c_groot_pp.h (a main header you should create).

Create Tensors for XOR X data and Y targets.

Build a Sequential model: Linear(2, 4), Tanh(), Linear(4, 1), Sigmoid().

Create MSELoss and SGD (or Adam).

Write the training loop (e.g., 1000 epochs):

optimizer.zero_grad()

auto y_pred = model.forward(X)

auto loss = criterion.forward(y_pred, Y)

loss.backward()

optimizer.step()

Print loss.*/