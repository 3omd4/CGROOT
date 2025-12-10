#ifndef _ACTIVATION_H
#define _ACTIVATION_H

#include <math.h>
#include <vector>

using namespace std;

// applies the sigmoid activation function
// input:        input (double value)
// output:       N/A
// side effect:  input is set with result value
// note:         N/A
void sigmoid_Funct(double &input);

// applies the Tanh activation function
// input:        input (double value)
// output:       N/A
// side effect:  input is set with result value
// note:         N/A
void tanh_Funct(double &input);

// applies the ReLU activation function
// input:        input (double value)
// output:       N/A
// side effect:  input is set with result value
// note:         N/A
void reLU_Funct(double &input);

// applies the derivative of the ReLU activation function
// input:        input (double value)
// output:       double value(the result)
// side effect:  N/A
// note:         N/A
double d_reLU_Funct(double input);

// applies the derivative of the sigmoid activation function
// input:        input (double value)
// output:       double value(the result)
// side effect:  N/A
// note:         N/A
double d_sigmoid_Funct(double input);

// applies the derivative of the Tanh activation function
// input:        input (double value)
// output:       double value(the result)
// side effect:  N/A
// note:         N/A
double d_tanh_Funct(double input);

#endif
