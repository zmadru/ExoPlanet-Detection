import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size=5, pool_size=3):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = out_channels  # Mantener la misma cantidad de canales para las capas siguientes
        layers.append(nn.MaxPool1d(pool_size, stride=1))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class ExoNet(nn.Module):
    def __init__(self, input_shapes, classes):
        super(ExoNet, self).__init__()
        
        self.global_par_blocks = nn.ModuleList()
        self.global_impar_blocks = nn.ModuleList()
        self.local_par_blocks = nn.ModuleList()
        self.local_impar_blocks = nn.ModuleList()
        
        for shape in input_shapes['global_par']:
            num_layers = shape[0]  # Asumiendo que el número de capas depende de la primera dimensión
            self.global_par_blocks.append(ConvBlock(1, 16 * 2 ** num_layers, num_layers))
        
        for shape in input_shapes['global_impar']:
            num_layers = shape[0]
            self.global_impar_blocks.append(ConvBlock(1, 16 * 2 ** num_layers, num_layers))
        
        for shape in input_shapes['local_par']:
            num_layers = shape[0]
            self.local_par_blocks.append(ConvBlock(1, 16 * 2 ** num_layers, num_layers))
        
        for shape in input_shapes['local_impar']:
            num_layers = shape[0]
            self.local_impar_blocks.append(ConvBlock(1, 16 * 2 ** num_layers, num_layers))
        
        self.batch_norm = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, len(classes))
    
    def forward(self, inputs):
        global_par, global_impar, local_par, local_impar = inputs
        
        global_par_out = torch.cat([block(x) for block, x in zip(self.global_par_blocks, global_par)], dim=-1)
        global_impar_out = torch.cat([block(x) for block, x in zip(self.global_impar_blocks, global_impar)], dim=-1)
        local_par_out = torch.cat([block(x) for block, x in zip(self.local_par_blocks, local_par)], dim=-1)
        local_impar_out = torch.cat([block(x) for block, x in zip(self.local_impar_blocks, local_impar)], dim=-1)
        
        x = torch.cat([global_par_out, global_impar_out, local_par_out, local_impar_out], dim=-1)
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(model, inputs, labels, epochs=10, batch_size=64, test_size=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    X_train, X_test, y_train, y_test = train_test_split(*inputs, labels, test_size=test_size, shuffle=False)
    
    X_train = [torch.tensor(x, dtype=torch.float32).to(device) for x in X_train]
    X_test = [torch.tensor(x, dtype=torch.float32).to(device) for x in X_test]
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")
