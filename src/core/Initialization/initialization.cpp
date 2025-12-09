#include "initialization.h"


//initializes an array with random values based on the number of inputs (and ouputs)
//input:        -arr (the array of values to be initialized)
//              -N (the number of inputs or inputs + outputs)
//              -distType (the type of probability distribution used)
//output:       N/A
//side effect:  fill arr with random values based on N and the distribution used
//Note:         this funciton is used as a base function for xavier and kaiming 
//              initialization functions
void init_Func(vector<double>& arr, size_t N, distributionType distType)
{
    //make random device and number generators to generate random numbers
    random_device rd;
    mt19937 gen(rd());

    //choose which distribution to use
    switch(distType)
    {

        case normalDistribution:
        {
            //in normal distribution, the distribution is made using the mean (0.0) and the varince
            double variance = sqrt(2.0/static_cast<double>(N)); //calculate the vairance 
            normal_distribution<double> dist(0.0, variance);  //make the distribution to be used
            for(size_t i = 0; i < arr.size(); i++)  //set every entry to random value
            {
                arr[i] = dist(gen);
            }
            return;
        }
        case uniformDistribution:
        {
            //in uniform distribution, the distribution is made using range of values(limits)
            double limit = sqrt(6.0/static_cast<double>(N));    //the limit is the variance*sqrt(3)
            uniform_real_distribution<double> dist(-limit, limit);  //make the distribution to be used
            for(size_t i = 0; i < arr.size(); i++)  //set every entry to random value
            {
                arr[i] = dist(gen);
            }
            return;
        }
    }
    
}

//initialize an array using kaiming He initialization function
//input:        -arr (the array of values to be initialized)
//              -numofInputs (the number of inputs)
//              -distType (the type of probability distribution used)
//output:       N/A
//side effect:  fill arr with random values based on N and the distribution used
//Note:         N/A
void init_Kaiming(vector<double>& arr, size_t numOfInputs, distributionType distType)
{
    init_Func(arr, numOfInputs, distType);
}

//initialize an array using Xavier Glorot initialization function
//input:        -arr (the array of values to be initialized)
//              -numOfInputs (the number of inputs)
//              -numOfOutputs (the number of outputs)
//              -distType (the type of probability distribution used)
//output:       N/A
//side effect:  fill arr with random values based on N and the distribution used
//Note          N/A
void init_Xavier(vector<double>& arr, size_t numOfInputs, size_t numOfOutputs, distributionType distType)
{
    init_Func(arr, numOfInputs + numOfOutputs, distType);
} 