import torch
import torch.nn as nn
import quantum_ops
import q_pars


class PatchQuantumGenerator(nn.Module):

    def __init__(self, n_generators, q_delta=1):

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_pars.q_depth * q_pars.n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        patch_size = 2 ** (q_pars.n_qubits - q_pars.n_a_qubits)

        images = torch.Tensor(x.size(0), 0).to(quantum_ops.device)

        for params in self.q_params:

            patches = torch.Tensor(0, patch_size).to(quantum_ops.device)
            for elem in x:
                q_out = quantum_ops.partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            images = torch.cat((images, patches), 1)

        return images
