#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <vector>
#include <math.h>
using namespace std;

//applies the sigmoid activation function
//input:        input (double value)
//output:       N/A
//side effect:  input is set with result value
//note:         N/A
void sigmoid_Funct(double& input);

//applies the Tanh activation function
//input:        input (double value)
//output:       N/A
//side effect:  input is set with result value
//note:         N/A
void tanh_Funct(double& input);

//applies the ReLU activation function
//input:        input (double value)
//output:       N/A
//side effect:  input is set with result value
//note:         N/A
void reLU_Funct(double& input);

//Calculates the derivative of the Sigmoid function
//input:        input (the pre-activated value 'z')
//output:       double (the derivative value)
//side effect:  N/A
//note:         Calculates sigmoid(z) * (1 - sigmoid(z))
double sigmoid_Prime(double input);

//Calculates the derivative of the Tanh function
//input:        input (the pre-activated value 'z')
//output:       double (the derivative value)
//side effect:  N/A
//note:         Calculates 1 - tanh(z)^2
double tanh_Prime(double input);

//Calculates the derivative of the ReLU function
//input:        input (the pre-activated value 'z')
//output:       double (1.0 if input > 0, else 0.0)
//side effect:  N/A
//note:         The derivative at exactly 0 is undefined, but treated as 0 here
double reLU_Prime(double input);

//Applies the Softmax activation function
//input:        input (vector of raw logits/scores)
//output:       N/A
//side effect:  input vector is modified in-place to contain probabilities
//note:         Uses "stability shift" (subtracting max) to prevent numerical overflow
void softmax_Funct(vector<double>& input);

#endif // _ACTIVATION_H
