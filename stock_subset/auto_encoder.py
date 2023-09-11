import torch
import torch.nn as nn
from typing import List, Optional
from quantum_layer.ortho_nn_butterfly import run_ortho_nn_with_unload
from quantum_layer.butterfly import num_butterfly_gates


class AutoEncoderHiddenLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        super(AutoEncoderHiddenLayer, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(
            self,
            vector: torch.Tensor,
    ):
        return self.feed_forward(vector)


class AutoEncoderOutputLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        super(AutoEncoderOutputLayer, self).__init__()
        self.feed_forward = nn.Linear(input_dim, output_dim)

    def forward(
            self,
            vector: torch.Tensor
    ):
        return self.feed_forward(vector)


class AutoEncoder(nn.Module):
    def __init__(
            self,
            dims: List[int]
    ):
        super(AutoEncoder, self).__init__()
        if len(dims) % 2 != 1:
            raise RuntimeError('layers_dims must be a vector of odd length')
        self.num_hidden_layers = int((len(dims) - 2))
        self.hidden_layers = [AutoEncoderHiddenLayer(dims[i], dims[i + 1]) for i in range(self.num_hidden_layers)]
        self.output_layer = AutoEncoderOutputLayer(dims[-2], dims[-1])

    def forward(
            self,
            vector: torch.Tensor
    ):
        for i in range(self.num_hidden_layers):
            vector = self.hidden_layers[i].forward(vector)

        output = self.output_layer.forward(vector)
        return output


class QuantumAutoEncoder(nn.Module):
    def __init__(
            self,
            dims: List[int],
    ):
        super(QuantumAutoEncoder, self).__init__()
        if len(dims) % 2 != 1:
            raise RuntimeError('layers_dims must be a vector of odd length')
        self.num_enc_layers = int((len(dims) - 1) / 2)
        self.enc_layers = [AutoEncoderHiddenLayer(dims[i], dims[i + 1]) for i in range(self.num_enc_layers)]
        self.dec_layers = [AutoEncoderHiddenLayer(dims[i], dims[i + 1]) for i in range(self.num_enc_layers, len(dims) - 2)]
        self.output_layer = AutoEncoderOutputLayer(dims[-2], dims[-1])
        self.num_dec_layers = len(self.dec_layers)

        num_onn_params = num_butterfly_gates(dims[self.num_enc_layers])
        self.onn_parameters = torch.randn(
            num_onn_params, requires_grad=True
        )
        self.onn_parameters = nn.Parameter(self.onn_parameters)

    def forward(
            self,
            vector: torch.Tensor
    ):
        for i in range(self.num_enc_layers):
            vector = self.enc_layers[i].forward(vector)

        vector = run_ortho_nn_with_unload(
            input_data=vector,
            params=self.onn_parameters,
            first_output_entry_positive=False,
        )

        for i in range(self.num_dec_layers):
            vector = self.dec_layers[i].forward(vector)

        output = self.output_layer.forward(vector)
        return output

