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
convLayer::convLayer(convKernels kernelConfig, activationFunction actFunc, 
                initFunctions initFunc, distributionType distType
                , featureMapDim FM_Dim)
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
convLayer::kernelType convLayer::initKernel(convKernels kernelConfig, initFunctions initFunc,
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
