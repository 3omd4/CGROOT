#include "model.h"
#include <math.h>
#include <random>
using namespace std;

random_device rd;
mt19937 gen(rd());

NNModel::NNModel(struct architecture, unsigned int numOfClasses)
{
    
}


void NNModel::intialization_He(vector<double>& arr, unsigned int numOfInputs)
{
    normal_distribution<double> dist(0.0, sqrt(2.0/static_cast<double>(numOfInputs)));
    for(unsigned int i = 0; i < arr.size(); i++)
    {
        arr[i] = dist(gen);
    }
}