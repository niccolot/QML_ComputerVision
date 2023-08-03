import pennylane as qml
import torch
import q_pars


# quantum device
dev = qml.device("lightning.qubit", wires=q_pars.q_pars.n_qubits)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_circuit(noise, weights):

    weights = weights.reshape(q_pars.q_depth, q_pars.q_pars.n_qubits)

    for i in range(q_pars.q_pars.q_pars.n_qubits):
        qml.RY(noise[i], wires=i)

    for i in range(q_pars.q_depth):
        for y in range(q_pars.n_qubits):
            qml.RY(weights[i][y], wires=y)

        for y in range(q_pars.n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(q_pars.n_qubits)))



def partial_measure(noise, weights):
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (q_pars.n_qubits - q_pars.n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven
