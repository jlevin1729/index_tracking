import torch
from quantum_layer.gates import rbs
from quantum_layer.butterfly_operator import apply_butterfly_operator


def apply_butterfly_rbs(
    params: torch.Tensor,
    butterfly_num_qubits: int,
    state: torch.Tensor,
):
    return apply_butterfly_operator(
        gates=rbs(params),
        butterfly_num_qubits=butterfly_num_qubits,
        state=state,
    )
