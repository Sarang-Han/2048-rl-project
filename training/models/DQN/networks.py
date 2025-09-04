import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class CNN2048Network(nn.Module):
    """2048용 CNN 네트워크"""
    
    def __init__(self, input_channels: int = 16, hidden_dim: int = 512):
        super(CNN2048Network, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        conv_output_size = 4 * 4 * 128
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He 초기화 적용"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # 차원 변환 (H, W, C) -> (C, H, W)
        if x.dim() == 4 and x.shape[-1] == 16:
            x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 네트워크"""
    
    def __init__(self, base_network: nn.Module):
        super(DuelingDQNNetwork, self).__init__()
        
        self.base_network = base_network
        
        # Dueling head
        hidden_dim = base_network.fc2.out_features
        base_network.fc3 = nn.Identity()
        
        # Value와 Advantage stream
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, 4)
        
        # 초기화
        nn.init.kaiming_normal_(self.value_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.value_head.bias, 0)
        nn.init.kaiming_normal_(self.advantage_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.advantage_head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_network(x)
        
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

def create_network(use_dueling: bool = False, **kwargs) -> nn.Module:
    """네트워크 생성 함수"""
    base_net = CNN2048Network(**kwargs)
    
    if use_dueling:
        return DuelingDQNNetwork(base_net)
    else:
        return base_net

def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)