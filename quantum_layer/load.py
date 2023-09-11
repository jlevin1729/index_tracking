from typing import Optional

import numpy as np
import torch
import unitair.states


def unary_loader_matrix(num_qubits, device: torch.device = torch.device("cpu")):
    """Get a matrix which implements unary loading.

    This matrix does not include the imaginary part.
    """
    loader = torch.zeros(2**num_qubits, num_qubits, device=device)
    for i in range(num_qubits):
        loader[2**i, num_qubits - i - 1] = 1
    return loader


def matrix_vector_mul(matrix: torch.Tensor, vector: torch.Tensor):
    """Apply a matrix to a batch of vectors."""
    vector = vector.unsqueeze(-1)  # This makes vector a "column"
    return (matrix @ vector).squeeze(-1)


def unary_load(
    data: torch.Tensor,
    loader_matrix: Optional[torch.Tensor] = None,
):
    """Load data into a unary matrix.

    Data is automatically normalized (with real L2 norm).

    Args:
        data: Tensor with size (*batch, num_features) which is expected
            to already be normalized. This data is normalized again by
            this function and so failing to normalize in advance may result
            in unexpected behavior.
        loader_matrix: When not given, the loader matrix is constructed.
            Otherwise, this function uses whatever loader matrix is given.
            The loader matrix must have size (2^num_features, num_features).
            Note that this matrix of real numbers because unary loading
            does not have an imaginary part.

    Returns: Tensor with size (*batch, 2, 2^num_features) giving the quantum
        state after loading.

    """
    #data = normalize_data(data)
    num_features = data.size()[-1]
    if loader_matrix is None:
        loader_matrix = unary_loader_matrix(num_qubits=num_features, device=data.device)
    real = matrix_vector_mul(loader_matrix, data)
    imag = torch.zeros_like(real)
    return torch.stack([real, imag], dim=-2)


def unary_unload(
    state: torch.Tensor,
    remove_imaginary_part: bool = True,
    loader_matrix: Optional[torch.Tensor] = None,
    first_entry_positive: bool = True,
):
    """Given a unary-loaded state, recover the original vector.

    This function is unphysical: the exact state vector is converted
    to a classical vector including phase. This does not simulate
    tomography error in any way.

    Args:
        state: State with size (*batch, 2, 2^n). This is assumed to
            be unary-encoded data meaning that there should be zeros everywhere
            except when the last index has the values 2^k for k = 0, ..., n-1.

        remove_imaginary_part: When True, the return state will not have
            a component for the imaginary part of the state.

        loader_matrix: The matrix which performs unary loading. There is no
            need to supply this argument, because when not given, matrix is
            constructed. Giving the argument is for caching since it's sensible
            to reuse many times for loading and unloading.
            The loader matrix must have size (2^num_features, num_features).
            Note that this matrix of real numbers because unary loading
            does not have an imaginary part.

        first_entry_positive: When True, we assume that the first
            entry is positive. This means that the unloaded vector may
            be minus the loaded vector.

    Returns: Tensor with size (*batch, num_qubits) or (*batch, 2, num_qubits)
        when `remove_imaginary_part` is True and False respectively.
    """
    if remove_imaginary_part:
        state = state.select(dim=-2, index=0)

    num_qubits = unitair.states.count_qubits(state)
    if loader_matrix is None:
        loader_matrix = unary_loader_matrix(num_qubits=num_qubits, device=state.device)

    out = matrix_vector_mul(loader_matrix.transpose(0, 1), state)
    if first_entry_positive:
        return force_first_entry_positive(out)
    else:
        return out


def force_first_entry_positive(unloaded_vector: torch.Tensor):
    """Force the first component of an unloaded vector to be positive.

    This function applies a -1 to every batch entry where the first
    component is negative. The reason we would ever use a function
    like this is because actual tomography cannot determine the
    overall sign, and we often deal with that by taking the
    first component to have phase 0.

    Note that this function is only meant to deal with the output
    from unary_unload. In other words, this function does NOT act on
    state vectors but rather on classical vectors to post-process.
    """
    first_entry = unloaded_vector.index_select(
        dim=-1, index=torch.tensor(0, device=unloaded_vector.device)
    )
    return first_entry.sign() * unloaded_vector


def normalize_data(data: torch.Tensor):
    """Get a normalized copy of given data.

    Args:
        data: Tensor with size (*batch, num_features)
    """
    norm = torch.linalg.norm(data, dim=-1)
    return data / norm.unsqueeze(-1)
