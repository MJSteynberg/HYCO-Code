import torch
import torch.nn as nn


class SpatioTemporalNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, activation='relu'):
        super(SpatioTemporalNN, self).__init__()
        
        # Input dimension is 3 (x, y, t)
        input_dim = 3
        # Output dimension is 1 (u)
        output_dim = 1
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Build the network layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y, t):
        """
        Forward pass of the network
        
        Args:
            x: spatial coordinate (batch_size, 1) or (batch_size,)
            y: spatial coordinate (batch_size, 1) or (batch_size,)
            t: temporal coordinate (batch_size, 1) or (batch_size,)
        
        Returns:
            u: predicted field value (batch_size, 1)
        """
        # Ensure inputs are 2D tensors
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Concatenate inputs
        inputs = torch.cat([x, y, t], dim=-1)
        
        # Forward pass through network
        u = self.network(inputs)
        
        return u