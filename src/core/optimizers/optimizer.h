#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

class Optimizer {
protected:
  double learning_rate;
  double weight_decay;

public:
  Optimizer(double lr, double wd = 0.0);
  virtual ~Optimizer() = default;
  virtual void update(std::vector<double> &weights,
                      const std::vector<double> &grads) = 0;
};

class SGD : public Optimizer {
public:
    SGD(double lr, double wd = 0.0);

    /** * Updates parameters using gradients.
     * * NOTE: The behavior depends on the input 'grads':
     * - If 'grads' is from 1 image -> Acts as SGD.
     * - If 'grads' is average of all images -> Acts as Batch GD.
     */
    void update(std::vector<double>& weights, const std::vector<double>& grads) override;
};

class SGD_Momentum: public Optimizer {
private:
  double momentum;
  std::vector<double> v; // Velocity for momentum
public:
  SGD_Momentum(double lr, double mom = 0.0, double wd = 0.0);

  /** * Updates parameters using gradients.
   * * NOTE: The behavior depends on the input 'grads':
   * - If 'grads' is from 1 image -> Acts as SGD.
   * - If 'grads' is average of all images -> Acts as Batch GD.
   */
  void update(std::vector<double> &weights,
              const std::vector<double> &grads) override;
};

class Adam : public Optimizer {
private:
  double beta1;   // Decay rate for momentum (default 0.9)
  double beta2;   // Decay rate for squared gradients (default 0.999)
  double epsilon; // Small number to prevent division by zero
  int t;          // Time step counter

  // These vectors hold the "memory" for every weight
  std::vector<double> m; // First moment (Momentum)
  std::vector<double> v; // Second moment (RMSprop)

public:
  // Standard Adam defaults: lr=0.001, b1=0.9, b2=0.999
  Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8,
       double wd = 0.0);

  void update(std::vector<double> &weights,
              const std::vector<double> &grads) override;
};

class RMSprop : public Optimizer {
private:
  double beta; // Decay rate
  double epsilon;
  std::vector<double> s; // Squared gradient moving average

public:
  RMSprop(double lr = 0.001, double b = 0.9, double eps = 1e-8,
          double wd = 0.0);
  void update(std::vector<double> &weights,
              const std::vector<double> &grads) override;
};

// Factory
struct OptimizerConfig; // Forward dec
Optimizer *createOptimizer(const OptimizerConfig &config);

#endif
