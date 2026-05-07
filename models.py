import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class TorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(100,), activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.apply(self._init_weights)  # Xavier initialization
	
        # Feature scaling
        self.scaler = StandardScaler()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                #m.bias.uniform_(-init_bound, init_bound)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class TorchLogistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.scaler = StandardScaler()

    def forward(self, x):
        return self.linear(x)


class TorchLogisticCNN(nn.Module):
    def __init__(self, num_img, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(3 * num_img, output_dim, 3, padding=0)
        with torch.no_grad():
            self.apply(self._init_weights)  # Xavier initialization
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)  #, gain=0.1)
            if m.bias is not None:
                #m.bias.uniform_(-init_bound, init_bound)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        #x = x.mean(dim=(2,3))
        x = x.amax(dim=(2,3))
        return x


class TorchCNN(nn.Module):
    def __init__(self, num_img, output_dim, img_size, num_conv_layers=2, conv_hidden_dim=64, num_fc_layers=1, fc_hidden_dim=128):
        super().__init__()
        self.img_size = img_size
        
        # Build convolutional layers
        conv_layers = []
        in_channels = 3 * num_img
        
        for i in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, conv_hidden_dim, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = conv_hidden_dim
        
        self.conv = nn.Sequential(*conv_layers)
        
        self.flatten_size = self._compute_flatten_size(num_img)
        print(f"flatten_size: {self.flatten_size}")
        
        # Build fully connected layers
        fc_layers = []
        prev_dim = self.flatten_size
        
        for i in range(num_fc_layers):
            fc_layers.append(nn.Linear(prev_dim, fc_hidden_dim))
            fc_layers.append(nn.ReLU())
            prev_dim = fc_hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)
        
        with torch.no_grad():
            self.apply(self._init_weights)
    
    def _compute_flatten_size(self, num_img):
        # Create dummy input to compute flattened size
        dummy_input = torch.zeros(1, 3 * num_img, self.img_size[0], self.img_size[1])
        dummy_output = self.conv(dummy_input)
        return int(np.prod(dummy_output.shape[1:]))
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x