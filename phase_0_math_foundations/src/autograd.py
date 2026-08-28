"""
This code implements a scalar-valued automatic differentiation (Autograd) engine from scratch.
It dynamically builds a computational graph as mathematical operations are performed on `Value` objects. 
By traversing this graph in reverse (reverse-mode autodiff), it can automatically compute the exact 
gradients of complex mathematical expressions and neural network losses with respect to their input parameters 
using the chain rule.
"""
import math

class Value:
    """
    A scalar value that tracks its computation history for reverse-mode autodiff.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        """
        Initializes a scalar Value. Sets its gradient to zero and stores its 
        children nodes to maintain the computational graph structure.
        """
        self.data = float(data)
        self.grad = 0.0
        # Internal function to chain the derivative using the chain rule
        self._backward = lambda: None
        # Track previous nodes to build the computational graph
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        """Returns a string representation of the Value object for debugging."""
        return f"Value(data={self.data})"

    def __add__(self, other):
        """
        Performs the forward pass for addition and defines the local derivative 
        function `_backward` to distribute gradients during the backward pass.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Local derivative of addition is 1.0. 
            # Global derivative = local derivative * out.grad
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        """Supports commutative addition (e.g., float + Value)."""
        return self + other

    def __mul__(self, other):
        """
        Performs the forward pass for multiplication and defines the local derivative 
        function `_backward` based on the product rule.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Local derivative of x*y with respect to x is y (and vice versa)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        """Supports commutative multiplication (e.g., float * Value)."""
        return self * other

    def __pow__(self, other):
        """
        Performs the forward pass for raising a Value to a constant power and 
        defines the local derivative function `_backward` based on the power rule.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        """Returns the negation of the Value."""
        return self * -1

    def __sub__(self, other):
        """Defines subtraction using addition and negation."""
        return self + (-other)

    def __rsub__(self, other):
        """Supports reversed subtraction (e.g., float - Value)."""
        return other + (-self)

    def __truediv__(self, other):
        """Defines division using multiplication and negative powers."""
        return self * other**-1

    def __rtruediv__(self, other):
        """Supports reversed division (e.g., float / Value)."""
        return other * self**-1

    def tanh(self):
        """
        Applies the hyperbolic tangent (tanh) activation function and defines 
        its local derivative for the backward pass.
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # Derivative of tanh(x) is 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """
        Applies the Rectified Linear Unit (ReLU) activation function and defines 
        its local subgradient for the backward pass.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # Derivative of ReLU is 1 if x is greater than 0, else 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Traverses the topological graph backwards to compute all gradients.
        It sorts nodes from output to inputs, applies the base gradient of 1.0 
        to the final output, and triggers the chain rule along the edges.
        """
        topo = []
        visited = set()
        
        # Build a topological sort of the graph
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # The base case: derivative of the final output with respect to itself is 1.0
        self.grad = 1.0
        
        # Apply the chain rule in reverse order
        for node in reversed(topo):
            node._backward()