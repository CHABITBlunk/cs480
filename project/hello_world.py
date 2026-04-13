import pennylane as qp
from functools import partial

dev = qp.device("default.qubit")

@qp.set_shots(5)
@qp.qnode(dev)
def bell_state():
    qp.H(wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.sample()

def uniform_superposition(n: int):
    for i in range(n):
        qp.H(i)

def oracle(value: int, n: int):
    ctrl_value = [(value >> i) & 1 for i in range(n)]
    qp.X('a')
    qp.H('a')
    qp.ctrl(qp.X, range(n), control_values=ctrl_value)
    qp.H('a')
    qp.X('a')

def diffuse(n: int):
    uniform_superposition(n)
    oracle(0, n)
    uniform_superposition(n)

value = 3

@qp.set_shots(1)
@qp.qnode(dev)
def grover(n: int, r: int):
    uniform_superposition(n)
    for _ in range(r):
        qp.FlipSign(value, wires=range(n))
        qp.GroverOperator(wires=range(n))

    return qp.sample(wires=range(n))

print(grover(5, 1))
