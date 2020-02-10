import torch.nn as nn
import torch
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size=(128, 128), activation='tanh', enc_dim=0):
        super().__init__()
        self.enc_dim = enc_dim
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.logic = nn.Linear(last_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        prob = torch.sigmoid(self.logic(x))
        
        if self.enc_dim < 2:
            return prob
        else:
            # Extract policy from sample
            p = x[-1:]
            eprob = np.exp(prob.detach().numpy())
            # See eq.5 in SGAIL paper
            return eprob/(eprob+p)
    