

#ifndef CGROOT_OPTIMIZER_H
#define CGROOT_OPTIMIZER_H


class Optimizer {
public:
    double learning_rate;

    Optimizer(double lr);
    virtual ~Optimizer() = default;

    // Pure virtual update function
    virtual void update(std::vector<double>& params, const std::vector<double>& grads) = 0;
};


// SGD Declaration

class SGD : public Optimizer {
public:
    SGD(double lr);

    // We only declare the function here. Implementation goes in .cpp
    void update(std::vector<double>& params, const std::vector<double>& grads) override;
};

// Adam Declaration

class Adam : public Optimizer {
private:
    double beta1;
    double beta2;
    double epsilon;
    int t; // Time step

    // Internal state
    std::vector<double> m;
    std::vector<double> v;

public:
    Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);

    void update(std::vector<double>& params, const std::vector<double>& grads) override;
};



#endif //CGROOT_OPTIMIZER_H