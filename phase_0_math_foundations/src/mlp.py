"""
This code implements a Multi-Layer Perceptron (MLP) neural network architecture 
from scratch, using the custom `Value` class from `autograd.py` as its foundation. 
It follows a hierarchical, object-oriented design similar to PyTorch's `nn.Module`, 
where individual Neurons make up Layers, and Layers make up the entire MLP. 
This structure allows for easy forward passes and parameter retrieval for optimization.
"""
import random
from src.autograd import Value

class Module:
    """Base class for all neural network modules."""
    def zero_grad(self):
        """
        Resets all gradients to 0. This is strictly required before starting 
        a new backward pass to prevent gradient accumulation from previous training steps.
        """
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        """Returns a list of all trainable parameters in the module."""
        return []

class Neuron(Module):
    """A single artificial neuron."""
    def __init__(self, nin, nonlin=True):
        """
        Initializes a neuron with `nin` random weights and one bias value.
        `nonlin` determines if a non-linear activation (tanh) will be applied.
        """
        # Initialize weights with random values between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Performs the forward pass of the neuron: calculates the dot product 
        of inputs `x` and weights `w`, adds the bias `b`, and applies activation.
        """
        # Computes: sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        """Returns the weights and bias of this specific neuron."""
        return self.w + [self.b]

class Layer(Module):
    """A fully connected neural network layer containing multiple neurons."""
    def __init__(self, nin, nout, nonlin=True):
        """
        Initializes the layer with `nout` independent neurons, each taking `nin` inputs.
        """
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x):
        """
        Computes the forward pass for the entire layer by passing the input `x` 
        through all of its neurons and collecting their outputs.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Collects and returns all parameters from all neurons within this layer."""
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP(Module):
    """A Multi-Layer Perceptron (MLP) neural network."""
    def __init__(self, nin, nouts):
        """
        Constructs the sequential network. `nin` is the input dimension, and `nouts` 
        is a list of layer sizes. The final layer typically omits the non-linearity.
        """
        sz = [nin] + nouts
        # The final layer does not typically use a non-linear activation for regression or logits
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Executes the forward pass of the entire network by sequentially passing 
        the input `x` through each layer to produce the final output.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Collects and returns all parameters from all layers in the entire network."""
        return [p for layer in self.layers for p in layer.parameters()]