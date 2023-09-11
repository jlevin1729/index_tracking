import enum
import itertools
from typing import Optional, Iterable
import numpy as np

import more_itertools


class TomographyVariant(str, enum.Enum):
    NO_OFFSET = "NO_OFFSET"
    OFFSET = "OFFSET"
    BARE = "BARE"


def num_butterfly_gates(num_qubits: int):
    """Get the total number of gates in a pyramidal circuit."""
    return int((num_qubits / 2) * np.log2(num_qubits))


def butterfly_gate_placements(num_qubits: int):
    """Get the gate placemente for butterfly circuit"""
    if np.log2(num_qubits) % 1 != 0:
        raise RuntimeError("Number of qubits needs to be a power of 2")

    """
    gate_placements={
    2:[[0, 1]],
    4:[[0, 2], [1, 3], [0, 1], [2, 3]],
    8:[[0, 4], [4, 6], [1, 5], [5, 7], [2, 6], [3, 7], [0, 2], [1, 3], [0, 1], [2, 3], [4, 5], [6, 7]],
    16:[[0, 8], [8, 12], [12, 14], [1, 9], [9, 13], [13, 15], [2, 10], [10, 14], [3, 11], [11, 15], [4, 12], [5, 13], [6, 14], [7, 15], [0, 4], [4, 6], [1, 5], [5, 7], [2, 6], [3, 7], [0, 2], [1, 3], [8, 10], [9, 11], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    }

    def get_butterfly_index(num_qubits):

        if num_qubits == 2:
            return np.array([[[0, 1]]])
        else:
            rbs_idxs = get_butterfly_index(num_qubits // 2)
            first = np.concatenate([rbs_idxs, rbs_idxs + num_qubits // 2], 1)
            last = np.arange(num_qubits).reshape(1, 2, num_qubits // 2).transpose(0, 2, 1)
            rbs_idxs = np.concatenate([first, last], 0)
            return rbs_idxs

    butterfly_idxs = get_butterfly_index(num_qubits)

    butterfly_idxs_return=[]
    temp_list = []

    for elm in butterfly_idxs:
        for elm2 in elm:
            for elm3 in elm2:
                temp_list.append(elm3)
            butterfly_idxs_return.append(temp_list)
            temp_list = []

    return butterfly_idxs_return
    """

    def get_butterfly_index(num_qubits):
        num_qubits = int(num_qubits)
        if num_qubits == 2:
            return np.array([[[0, 1]]])
        else:
            rbs_idxs = get_butterfly_index(num_qubits // 2)
            first = np.concatenate([rbs_idxs, rbs_idxs + num_qubits // 2], 1)
            last = (
                np.arange(num_qubits).reshape((1, 2, num_qubits // 2)).transpose(0, 2, 1)
            )
            rbs_idxs = np.concatenate([first, last], 0)
            return rbs_idxs

    butterfly_idxs = get_butterfly_index(num_qubits)

    rbs_idxs = [[list(q) for q in qubits] for qubits in butterfly_idxs]
    return rbs_idxs[::-1]


def tomography_qubits(num_qubits):
    no_offset_index_range = num_qubits // 2
    offset_index_range = num_qubits // 2
    if num_qubits % 2 == 0:
        offset_index_range -= 1

    no_offset = [[2 * i, 2 * i + 1] for i in range(no_offset_index_range)]
    offset = [[2 * i + 1, 2 * i + 2] for i in range(offset_index_range)]

    return {TomographyVariant.NO_OFFSET: no_offset, TomographyVariant.OFFSET: offset}


if __name__ == '__main__':
    #gates = butterfly_gate_placements(16)
    num = num_butterfly_gates(16)
    print(num)
