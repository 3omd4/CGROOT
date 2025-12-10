#include "activation.h"

//applies the sigmoid activation function
//input:        input (double value)
//output:       N/A
//side effect:  input is set with result value
//note:         N/A
void sigmoid_Funct(double& input)
{
    input = 1.0/(1.0 + exp(-input));
}

//applies the Tanh activation function
//input:        input (double value)
//output:       N/A
//side effect:  input is set with result value
//note:         N/A
void tanh_Funct(double& input)
{
    input = 2.0/(1.0 + exp(-2.0 * input)) - 1.0;
}


//applies the ReLU activation function
//input:        input (double value)
//output:       N/A
//side effect:  input is set with result value
//note:         N/A
void reLU_Funct(double& input)
{
    input = (input > 0.0)? input : 0.0;
}

//applies the derivative of the ReLU activation function
//input:        input (double value)
//output:       double value(the result)
//side effect:  N/A
//note:         N/A
double d_reLU_Funct(double input)
{
    return (input > 0)? 1.0: 0.0;
}

//applies the derivative of the sigmoid activation function
//input:        input (double value)
//output:       double value(the result)
//side effect:  N/A
//note:         N/A
double d_sigmoid_Funct(double input)
{
    return input*(1.0 - input);
}

//applies the derivative of the Tanh activation function
//input:        input (double value)
//output:       double value(the result)
//side effect:  N/A
//note:         N/A
double d_tanh_Funct(double input)
{
    return 1.0 - input*input;
}
