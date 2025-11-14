#ifndef _MODEL_H
#define _MODEL_H


#include <random>
#include "../layers/layers.h"

typedef vector<vector<unsigned char>> image;

enum activationFunction{
    RelU,
    Sigmoid,
    Tanh,
    Softmax,
    NumOfActivationFunction,
};

//the struct that determine the arhitecture of the model
//which is used only to initialize the model
struct architecture
{
    unsigned int numOfLayers;
    int* neuronsPerLayer; 
    activationFunction* layerTypes;
    //the number of neurons per layer, in other words
    //the size of the output vector of each layer

    //additional data used to initialize the data
};



class NNModel
{
    private:
    Layer* firstLayer;
    image data;

    public:
    //make an array of layers and specify the number of neurons of each layer and the type of each layer
    //specify the activation and initialization functions
    //initialize the kernel and weights of each layer(initalize/construct each layer)
    NNModel(struct architecture, unsigned int numOfClasses);
    /*construction and initialization helper functions*/

    //function to calculate the number of neurons of the conv layer

    /*******************end helpers*******************/

    //this function takes the input image to train the model
    void train(image data, int trueOutput);

    //this function is used after the model is trained 
    //to either validate the model or use it
    //returns an int based on the classification
    int classify(image data);

    //initialization function here

    /*He initialization function*/
    void intialization_He(vector<double>& arr, unsigned int numOfInputs);



    /*Xavier intialization function*/


    //additional functions
    
};

#endif