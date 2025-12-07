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

//Calculates the derivative of the Sigmoid function
//input:        input (the pre-activated value 'z')
//output:       double (the derivative value)
//side effect:  N/A
//note:         Calculates sigmoid(z) * (1 - sigmoid(z))
double sigmoid_Prime(double input) {
    // f'(x) = f(x) * (1 - f(x))
    double sig = 1.0 / (1.0 + exp(-input)); 
    return sig * (1.0 - sig);
}

//Calculates the derivative of the Tanh function
//input:        input (the pre-activated value 'z')
//output:       double (the derivative value)
//side effect:  N/A
//note:         Calculates 1 - tanh(z)^2
double tanh_Prime(double input) {
    // f'(x) = 1 - f(x)^2
    double t = tanh(input);
    return 1.0 - t * t;
}

//Calculates the derivative of the ReLU function
//input:        input (the pre-activated value 'z')
//output:       double (1.0 if input > 0, else 0.0)
//side effect:  N/A
//note:         The derivative at exactly 0 is undefined, but treated as 0 here
double reLU_Prime(double input) {
    return (input > 0.0) ? 1.0 : 0.0;
}

//Applies the Softmax activation function
//input:        input (vector of raw logits/scores)
//output:       N/A
//side effect:  input vector is modified in-place to contain probabilities
//note:         Uses "stability shift" (subtracting max) to prevent numerical overflow
void softmax_Funct(vector<double>& input) {
    double max = -1e9;
    for(double d : input) if(d > max) max = d; 

    double sum = 0.0;
    for(size_t i=0; i<input.size(); i++) {
        input[i] = exp(input[i] - max);     
        sum += input[i];
    }
    for(size_t i=0; i<input.size(); i++) {
        input[i] /= sum;
    }
}
