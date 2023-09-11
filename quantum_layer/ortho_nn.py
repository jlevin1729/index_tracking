from typing import Optional

import torch
from pyramidal_rbs import apply_pyramidal_rbs
from load import unary_load, unary_unload


def run_ortho_nn_with_unload(
    input_data: torch.Tensor,
    params: torch.Tensor,
    pyramid_end: int = 0,
    loader_matrix: Optional[torch.Tensor] = None,
    first_output_entry_positive: bool = True,
):
    """Apply a single quantum orthogonal nn to given input data and unload.

    This function applies a unary loader to the input data and then
    applies one pyramidal RBS circuit layer. The final state is then
    unloaded. Note that true tomography is not modeled--loading and unloading
    is performed exactly by hand.

    However, we do model one aspect of tomography. When
    first_output_entry_positive is True, we set the first component
    of outputs to be positive. This is done with physical tomography.
    """
    state = ortho_nn_single_apply(
        input_data=input_data,
        params=params,
        pyramid_end=pyramid_end,
        loader_matrix=loader_matrix,
    )
    return unary_unload(
        state=state,
        remove_imaginary_part=True,
        loader_matrix=loader_matrix,
        first_entry_positive=first_output_entry_positive,
    )


def ortho_nn_single_apply(
    input_data: torch.Tensor,
    params: torch.Tensor,
    pyramid_end: int = 0,
    loader_matrix: Optional[torch.Tensor] = None,
):
    """Apply a single quantum orthogonal nn to given input data.

    This function applies a unary loader to the input data and then
    applies one pyramidal RBS circuit layer. The output is the quantum
    state at the end. This function does not perform tomography.

    Args:
        input_data: Size (*batch, num_qubits),
        params: Size (num_pyramid_params,)
        pyramid_end:
        loader_matrix:
    """
    num_features = input_data.size()[-1]
    state = unary_load(
        data=input_data,
        loader_matrix=loader_matrix,
    )
    return apply_pyramidal_rbs(
        params=params,
        pyramid_num_qubits=num_features,
        state=state,
        pyramid_end=pyramid_end,
    )
