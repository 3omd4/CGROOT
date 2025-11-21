#ifndef _INITIALIZATION_H
#define _INITIALIZATION_H

#include <cmath>
#include <random>
#include "../definitions.h"
using namespace std;


//initializes an array with random values based on the number of inputs (and ouputs)
//input:        -arr (the array of values to be initialized)
//              -N (the number of inputs or inputs + outputs)
//              -distType (the type of probability distribution used)
//output:       N/A
//side effect:  fill arr with random values based on N and the distribution used
//Note:         this funciton is used as a base function for xavier and kaiming 
//              initialization functions
void init_Func(vector<double>& arr, size_t N, distributionType distType);

//initialize an array using kaiming He initialization function
//input:        -arr (the array of values to be initialized)
//              -numofInputs (the number of inputs)
//              -distType (the type of probability distribution used)
//output:       N/A
//side effect:  fill arr with random values based on N and the distribution used
//Note:         N/A
void init_Kaiming(vector<double>& arr, size_t numOfInputs, distributionType distType);

//initialize an array using Xavier Glorot initialization function
//input:        -arr (the array of values to be initialized)
//              -numOfInputs (the number of inputs)
//              -numOfOutputs (the number of outputs)
//              -distType (the type of probability distribution used)
//output:       N/A
//side effect:  fill arr with random values based on N and the distribution used
//Note          N/A
void init_Xavier(vector<double>& arr, size_t numOfInputs, size_t numOfOutputs, distributionType distType);

#endif