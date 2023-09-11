import torch
from unitair.gates.gate_construction import parameterized_gate
from typing import Union


@parameterized_gate
def rbs(angle: Union[torch.Tensor, float]):
    """Get the 'reconfigurable beam splitter' gate.

    In the computational basis, this gate is the operator

    1    0     0     0
    0    cos   sin   0
    0   -sin   cos   0
    0    0     0     1

    where cos and sin are based on the given parameter `angle`.

    PyTorch device is inherited from the device of `angle`. If `angle` is
    a float, CPU is used.

    Args:
        angle: Tensor with size (batch_length,) or just ().

    Returns: Tensor with size (batch_length, 2, 4, 4) or (2, 4, 4) if there
        is no batch dimension. The (2, 4, 4) is such that the first dimension
        means the real and imaginary parts and the last two dimension are
        the matrices of the real an imaginary parts of the gates.
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones(angle.size(), device=angle.device)
    zero = torch.zeros(angle.size(), device=angle.device)

    return [
        [
            [one, zero, zero, zero],
            [zero, cos, sin, zero],
            [zero, -sin, cos, zero],
            [zero, zero, zero, one],
        ],
        [
            [zero, zero, zero, zero],
            [zero, zero, zero, zero],
            [zero, zero, zero, zero],
            [zero, zero, zero, zero],
        ],
    ]
