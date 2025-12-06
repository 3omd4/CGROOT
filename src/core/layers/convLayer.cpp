#include "layers.h"
#include "../Initialization/initialization.h"


//the convolution layer constructor
//input:        -kernelConfig (contains all the information about the kernel)
//              -actFunc (activation function)
//              -initFunc (initialization function)
//              -distType (distribution type)
//              -FM_Dim (the dimension of the output feature map)
//ouput:        N/A
//side effect:  the convolution layer is constructed
//Note:         N/A
convLayer::convLayer(convKernels& kernelConfig, activationFunction actFunc, 
                initFunctions initFunc, distributionType distType
                , featureMapDim& FM_Dim)
    : kernel_info(kernelConfig), fm(FM_Dim), act_Funct(actFunc)
{

    //make and initialize each kernel and store them in kernels vector
    for(size_t i = 0; i < kernelConfig.numOfKerenels; i++)
    {
        kernels.emplace_back(initKernel(kernelConfig, initFunc, distType));
    }
    
    //iterate and make each each feature map
    for(size_t i = 0; i < fm.FM_depth; i++)
    {
        featureMapType map(fm.FM_height);       //make a feature map
        for(size_t j = 0; j < fm.FM_height; j++)
        {
            map[j].assign(fm.FM_width, 0.0);    //make and initialzie each row to zeros
        }
        featureMaps.emplace_back(map);  //store the feature map in the feature maps vector
    }
}


//initialize a kernel
//input:        -kernelConfig (contains all the information about the kernel)
//              -initFunc (the initialization function)
//              -distType (the type of the distribution)
//output:       kernelType (the initialized kernel)
//side effect:  N/A
//Note:         N/A
convLayer::kernelType convLayer::initKernel(convKernels& kernelConfig, initFunctions initFunc,
                 distributionType distType)
{
    //make a 3D kernel
    kernelType k(kernelConfig.kernel_depth); //make a vector of 2D kernels

    //iterate over each 2D kernel
    for(size_t i = 0; i < k.size(); i++)
    {
        //make the 2D kernel
        k[i].resize(kernelConfig.kernel_height);

        //iterate over each row of the kernel
        for(size_t j = 0; j < k[i].size(); j++)
        {
            //make the row of the kernel and initialize it to zeros, which will be later 
            //initialized to random variables
            k[i][j].assign(kernelConfig.kernel_width, 0.0);

            //calculate n_in, the number of inputs, which is used in the initialization functions
            size_t n_in = kernelConfig.kernel_depth*kernelConfig.kernel_height*kernelConfig.kernel_width;
            switch(initFunc)    //choose which initialization function to use
            {
            case Kaiming:
                init_Kaiming(k[i][j], n_in, distType);
                break;
            case Xavier:
                //calculate n_out, the number of outputs to be used by Xavier initialization function
                size_t n_out = kernelConfig.numOfKerenels*kernelConfig.kernel_height*kernelConfig.kernel_width;
                init_Xavier(k[i][j], n_in, n_out, distType);
                break;
            }
        }
    }

    //return the initialized kernel
    return k;
}



//do the convolution operation by sweeping the kernels through 
//the input feature map and putin the result in the (output) feature map
//essentially doing the forward propagation
//input:        inputFeatureMaps (previous layer output feature maps)
//output:       N/A
//side effect:  this layer (output) feature maps is filled
//Note:         N/A
void convLayer::convolute(vector<featureMapType>& inputFeatureMaps)
{
    //do the same convolution operation using every kernel where each kernel
    //will result in a different feature map, iterate using "krnl"
    for(size_t krnl = 0; krnl < kernel_info.numOfKerenels; krnl++)
    {

        //iterate on every channel (depth) using 'd', where 'd' is the depth corrisponding
        //of the kernel depth and the input feature map depth
        for(size_t d = 0; d < kernel_info.kernel_depth; d++)
        {
            //moving on the 2D feature map the length to be moved is fm_dim - krnl_dim + 1

            //move along the rows of the input feature map using 'i'
            for(size_t i = 0; i < (inputFeatureMaps[d].size() - kernel_info.kernel_height + 1); i++)
            {
                //move along the columns of the input feature map using 'j'
                for(size_t j = 0; j < (inputFeatureMaps[d][i].size() - kernel_info.kernel_width + 1); j++)
                {
                    //do the convolution
                    //"k1" is row iterator and "k2" is the column iterator
                    for(size_t k1 = i; k1 < (i + kernel_info.kernel_height); k1++)
                    {
                        for(size_t k2 = j; k2 < (j + kernel_info.kernel_width); k2++)
                        {
                            //the convolutio is done by doing element-wise multiplication of 
                            //the input feature map and the kernel infront of this part of the feature map
                            //this is done for every channel of the input feature map and then stored in the
                            //corrisponding entry in the output feature map

                            //"krnl" indexes which kernel and its output feature map
                            //'d' is the depth of both the kernel and the input feature map
                            //"k1" and "k2" are the indexs of the row and column of which the the kernel
                            //is positioned, respectively, thsi position is the top-left corner of the kernel
                            //the result of the differnt channels of the same kernel are added to the corrisponding
                            //entry in the output feature map when the depth iterator 'd' is changed
                            featureMaps[krnl][i][j] += inputFeatureMaps[d][k1][k2]*kernels[krnl][d][k1-i][k2-j];
                        }
                    }


                }
            }
        }


    }
}



//do the forward propagation of the convolution layer
//by first applying the convolution and then the activation functions
//input:        inputFeatureMaps
//output:       N/A
//side effect:  the feature maps are filled with the forward propagation values
//note:         N/A
void convLayer::forwardProp(vector<featureMapType>& inputFeatureMaps)
{
    //apply the convolution
    convolute(inputFeatureMaps);

    //apply the activation function to every element of the output feature map
    for(size_t d = 0; d < fm.FM_depth; d++)
    {
        for(size_t h = 0; h < fm.FM_height; h++)
        {
            for(size_t w = 0; w < fm.FM_width; w++)
            {
                switch(act_Funct)
                {
                case RelU:
                    reLU_Funct(featureMaps[d][h][w]);
                    break;
                case Sigmoid:
                    sigmoid_Funct(featureMaps[d][h][w]);
                     break;
                case Tanh:
                    tanh_Funct(featureMaps[d][h][w]);
                    break;
                }
            }
        }
    }
}