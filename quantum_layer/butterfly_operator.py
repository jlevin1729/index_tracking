from typing import Optional

import torch
import unitair.simulation as sim
from quantum_layer.butterfly import butterfly_gate_placements, num_butterfly_gates


def apply_butterfly_operator(
    gates: torch.Tensor,
    butterfly_num_qubits: int,
    state: torch.Tensor,
):
    validate_butterfly_gates(
        gates=gates, num_qubits=butterfly_num_qubits
    )
    gate_placements = butterfly_gate_placements(
        num_qubits=butterfly_num_qubits
    )
    gate_iter = iter(gates)
    for layer_indices in gate_placements:
        for qubits in layer_indices:
            g = next(gate_iter)
            state = sim.apply_operator(operator=g, qubits=qubits, state=state)
    return state


def validate_butterfly_gates(
    gates: torch.Tensor,
    num_qubits: int,
):
    num_gates = num_butterfly_gates(num_qubits=num_qubits)
    if gates.size() != torch.Size([num_gates, 2, 4, 4]):
        raise RuntimeError(
            f"Expected gates to have size {(num_gates, 2, 4, 4)}."
            f"Found {tuple(gates.size())}."
        )
