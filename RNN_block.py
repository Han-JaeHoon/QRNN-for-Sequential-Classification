import pennylane as qml
class RNN_block:
    def __init__(self, num_of_qubits):
        self.num_of_qubits = num_of_qubits

    
    def embedding(self, params):
        n = self.num_of_qubits
        for i in range(n):
            qml.Hadamard(i)
            qml.RZ(2.0 * params[:,i], i)
        
        for i in range(n - 1):
            qml.IsingZZ(2.0 * params[:,n + i] ,[i , i + 1])
    
    def ansatz(self, params, all_entangled = False):
        # Length of Params : 3 * num_qubit
        n = self.num_of_qubits
        for i in range(n):
            qml.RX(params[3 * i], i)
            qml.RY(params[3 * i + 1], i)
            qml.RZ(params[3 * i + 2], i)
        for i in range(n - 1):
            qml.CNOT([i, i + 1])
        if all_entangled:
            qml.CNOT([n - 1, 0])