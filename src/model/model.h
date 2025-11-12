#ifndef _MODEL_H
#define _MODEL_H

#include "../layers/layers.h"

typedef vector<vector<unsigned char>> image;

//the struct that determine the arhitecture of the model
//which is used only to initialize the model
struct architecture
{
    unsigned int numOfLayers;
    int* neuronsPerLayer; 
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
    NNModel(struct architecture);

    //this function takes the input image to train the model
    void train(image data, int trueOutput);

    //this function is used after the model is trained 
    //to either validate the model or use it
    //returns an int based on the classification
    int detect(image data);

    //initialization function here

    //additional functions
    
};

#endif