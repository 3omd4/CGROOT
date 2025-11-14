//This file includes the layer abstract class declaration
//which every other layer class inherets from
//And the declaration of all other layers which include:
//input layer
//convolution layer
//activataion layer
//pooling layer
//output layer


#include <vector>
using namespace std;
#include "../model/model.h"


#ifndef _LAYERS_H
#define _LAYERS_H

//the data type of all vectors
typedef vector<double> dataType;


//this is the base class which every other layer class inherets from
class Layer
{
    private:    
    dataType toNextLayer;
    //the vector containing the data to be used by the next layer
    //the data of the vector is written at every iteration/every data entered

    public:
    Layer(unsigned int outputVectorSize);

    //these fucntions do forward and backward propagation 
    //and must be implemented differently at every layer
    //(the declaration probably isn't correct)
    virtual void forwardProp() = 0; 
    virtual void backwardProp() = 0;



        //additional functions

    

};


class inputLayer : public Layer
{
    private:
    //any additional data

    public:
    inputLayer(unsigned int outputVectorSize);

    //at the start of training this fucntion is 
    //used to make the data ready for the other layers
    void start(image data);

    //this function flattens the data for ease of use
    //and optimzaton purposes later
    void flatten();
    //additional functions


};

class convLayer : public Layer
{
    private:
    dataType kernel;
    //any additional data

    public:
    convLayer(unsigned int outputVectorSize, dataType KerenlInitValues);

    //this function do convolution on the data using
    //kerenl and stores the result in the toNextLayer vector
    void convolute();

    //additional functions
};

class activationLayer : public Layer
{
    private:
    dataType weights;
    //any additional data

    public:
    activationLayer(unsigned int outputVectorSize, dataType weightsInitValues);


    //activation functions

    //additional functions

};

class poolingLayer : public Layer
{
    private:
    //any additional data

    public:
    poolingLayer(unsigned int outputVectorSize);

    //pooling functions

    //additional functions
};

class outputLayer : public Layer
{
    private:
    //any additional data

    public:
    outputLayer(unsigned int outputVectorSize);

    //additional functions
};


#endif