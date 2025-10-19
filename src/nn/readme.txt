Provides the high-level, user-friendly API for building models. This is what your users will interact with.

module.h: Defines the nn::Module base class. This is the "Lego brick" base. It provides a parameters() method to gather all learnable weights.

parameter.h: Defines the nn::Parameter class. This is a wrapper around a Tensor that flags it as a learnable weight (i.e., requires_grad = true).