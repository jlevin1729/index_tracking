from typing import Optional

import torch
import unitair.simulation as sim
from logistics.circuits.pyramidal import pyramid_gate_placements, num_pyramid_gates


def apply_pyramidal_operator(
    gates: torch.Tensor,
    pyramid_num_qubits: int,
    state: torch.Tensor,
    pyramid_end: int = 0,
):
    validate_pyramidal_gates(
        gates=gates, num_qubits=pyramid_num_qubits, end=pyramid_end
    )
    gate_placements = pyramid_gate_placements(
        num_qubits=pyramid_num_qubits, end=pyramid_end
    )
    gate_iter = iter(gates)
    for layer_indices in gate_placements:
        for qubits in layer_indices:
            g = next(gate_iter)
            state = sim.apply_operator(operator=g, qubits=qubits, state=state)
    return state


def validate_pyramidal_gates(
    gates: torch.Tensor,
    num_qubits: int,
    end: int = 0,
):
    num_gates = num_pyramid_gates(num_qubits=num_qubits, end=end)
    if gates.size() != torch.Size([num_gates, 2, 4, 4]):
        raise RuntimeError(
            f"Expected gates to have size {(num_gates, 2, 4, 4)}."
            f"Found {tuple(gates.size())}."
        )
