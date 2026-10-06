import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autograd import Value
from mlp import MLP
from optimizers import SGD, SGDMomentum, RMSprop, Adam


def test_autograd_basic_ops():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f
    L.backward()

    # dL/dd = f = -2.0
    # dL/de = dL/dd * dd/de = -2.0 * 1 = -2.0
    # dL/da = dL/de * de/da = -2.0 * b = -2.0 * -3.0 = 6.0
    # dL/db = dL/de * de/db = -2.0 * a = -2.0 * 2.0 = -4.0
    assert a.grad == 6.0, f"Expected 6.0, got {a.grad}"
    assert b.grad == -4.0, f"Expected -4.0, got {b.grad}"
    assert c.grad == -2.0, f"Expected -2.0, got {c.grad}"


def test_autograd_branching():
    # Test that gradient accumulation works when a variable is reused: f(x) = x^2 + x^3
    x = Value(2.0)
    y = x ** 2 + x ** 3
    y.backward()
    # dy/dx = 2*x + 3*x^2 = 2*2 + 3*4 = 4 + 12 = 16
    assert x.grad == 16.0, f"Expected 16.0, got {x.grad}"


def test_autograd_activations():
    x = Value(0.5)
    t = x.tanh()
    t.backward()
    # dt/dx = 1 - tanh(x)^2
    expected_tanh_grad = 1.0 - (t.data ** 2)
    assert abs(x.grad - expected_tanh_grad) < 1e-6

    # Test ReLU positive and negative
    x_pos = Value(2.0)
    r_pos = x_pos.relu()
    r_pos.backward()
    assert x_pos.grad == 1.0

    x_neg = Value(-2.0)
    r_neg = x_neg.relu()
    r_neg.backward()
    assert x_neg.grad == 0.0


def test_mlp_forward_and_params():
    mlp = MLP(nin=3, nouts=[4, 4, 1])
    # Nin=3 -> Layer 1 (4 neurons): 4*(3 weights + 1 bias) = 16
    # Layer 2 (4 neurons): 4*(4 weights + 1 bias) = 20
    # Layer 3 (1 neuron): 1*(4 weights + 1 bias) = 5
    # Total = 16 + 20 + 5 = 41 parameters
    assert len(mlp.parameters()) == 41

    out = mlp([Value(1.0), Value(2.0), Value(3.0)])
    assert isinstance(out, Value)


def test_optimizers_quadratic_minimization():
    # Minimize f(w) = (w - 5)^2 using all four optimizers
    # Target w* = 5.0
    for opt_cls in [SGD, SGDMomentum, RMSprop, Adam]:
        w = Value(0.0)
        opt = opt_cls([w], lr=0.1)

        for _ in range(200):
            opt.zero_grad()
            loss = (w - 5.0) ** 2
            loss.backward()
            opt.step()

        assert abs(w.data - 5.0) < 0.25, f"{opt_cls.__name__} failed to converge: w = {w.data}"
