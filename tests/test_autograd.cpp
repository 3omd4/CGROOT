/*
Purpose: Unit test gradient calculations.

To-Do:

TEST(Autograd, Add): a(2, req_grad), b(3, req_grad). c = a + b. c.backward(). ASSERT_EQ(a.grad(), 1.0).

TEST(Autograd, Mul): a(2), b(3). c = a * b. c.backward(). ASSERT_EQ(a.grad(), 3.0). ASSERT_EQ(b.grad(), 2.0).

TEST(Autograd, NumericalGradCheck): This is the most important test. Write a function that numerically estimates the gradient ((f(x+h) - f(x-h)) / (2*h)) and asserts that it's very close to the gradient computed by backward(). Run this check for every single operation.
*/