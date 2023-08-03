import cirq
import numpy as np
import sympy


def one_qubit_rotation(qubit, symbols):
    """
    returns list of gates that rotate the single qubit
    by angle \vect{theta} = (symbols[0], symbols[1], symbols[2]) 
    """

    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]


def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on the qubits list 
    entangling sequentially the quibits in the list and 
    finally the first with the last like 

    [cz(q0,q1), cz(q1,q2), ... cz(qn-1,qn), cz(qn,q0)]
    """

    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])

    return cz_ops


def generate_circuit(qubits, n_layers, input_features, use_entanglement=False, use_terminal_entanglement=False):
    """
    Prepares a data re-uploading circuit on `qubits` with `n_layers` layers

    args: 
        qubits: (int) number of qubits desired

        n_layers: (int) number of layers desired

        input_features: (int) number of features of the data 
                        e.g. 3x3 images has 9 features

        use_entanglement: (bool) if to insert an entanglement layer between the 
                          variational gates
        
        use_terminal_entanglement: (bool) if to use entanglement layer at the end of
                                   the circuit

    
    returns: the data reuploading circuit and a list of the variational parameters 
    """

    n_qubits = len(qubits)

    # Zero-pad the inputs and params if it is not a multiple of 3
    padding = (3 - (input_features % 3)) % 3

    # Sympy symbols for weights and bias parameters
    params = sympy.symbols(f'theta(0:{(input_features + padding) * n_layers * n_qubits})')
    params = np.asarray(params).reshape((n_layers, n_qubits, (input_features + padding)))

    circuit = cirq.Circuit()
    for l in range(n_layers):
        for gate in range(int(np.ceil(input_features/3))):

            # Variational layer
            circuit += cirq.Circuit(
                one_qubit_rotation(q, params[l, i, gate * 3:(gate + 1) * 3]) for i, q in enumerate(qubits))

        # Entangling layer
        if n_qubits >= 2 and (l != (n_layers - 1) or n_layers == 1) and use_entanglement:
            circuit += entangling_layer(qubits)
        if n_qubits >= 2 and use_terminal_entanglement and (l == (n_layers - 1) and n_layers != 1):
            circuit += entangling_layer(qubits)

    return circuit, list(params.flat)