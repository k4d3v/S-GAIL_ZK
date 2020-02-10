import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh', encode_dim=0):
        super().__init__()
        self.encode_dim = encode_dim
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        
        if encode_dim > 1:
            # Define hidden layer for encode (See SGAIL paper)
            self.enc_layers = nn.ModuleList()
            self.enc_layers.append(nn.Linear(encode_dim, 128))


        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        """
        Forward pass for a n_samples x n_dims matrix.
        Arguments:
            x [Tensor] -- The matrix
        
        Returns:
            [Tensor] -- After forprop
        """
        if self.encode_dim < 2:
            for affine in self.affine_layers:
                x = self.activation(affine(x))

        # See net structure in paper
        # TODO: How to update value fun??
        else:
            s = x[:, :-self.encode_dim]
            e = x[:, -self.encode_dim:]
            # First layer
            s = self.activation(self.affine_layers[0](s))
            e = self.activation(self.enc_layers[0](e))
            # Concatenate
            se = s.add(e)
            # Final layer
            x = self.activation(self.affine_layers[1](se))

        return self.value_head(x)
