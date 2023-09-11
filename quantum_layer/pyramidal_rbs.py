import torch
from logistics.circuits.emulation.gates import rbs
from logistics.circuits.emulation.pyramidal_operator import apply_pyramidal_operator


def apply_pyramidal_rbs(
    params: torch.Tensor,
    pyramid_num_qubits: int,
    state: torch.Tensor,
    pyramid_end: int = 0,
):
    return apply_pyramidal_operator(
        gates=rbs(params),
        pyramid_num_qubits=pyramid_num_qubits,
        state=state,
        pyramid_end=pyramid_end,
    )
