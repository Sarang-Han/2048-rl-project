import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class CNN2048Network(nn.Module):
    """2048용 CNN 네트워크 - Layered 관찰 타입용"""
    
    def __init__(self, input_channels: int = 16, hidden_dim: int = 512):
        super(CNN2048Network, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate the size after convolutions (4x4 remains 4x4 with padding=1)
        conv_output_size = 4 * 4 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4 actions
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """신경망 가중치 초기화"""
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
        # Input shape: (batch_size, 4, 4, 16) -> (batch_size, 16, 4, 4)
        if x.dim() == 4 and x.shape[-1] == 16:
            x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten for fully connected layers - 수정된 부분
        x = torch.flatten(x, start_dim=1)  # view() 대신 flatten() 사용
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class DNN2048Network(nn.Module):
    """2048용 DNN 네트워크 - Flat 관찰 타입용"""
    
    def __init__(self, input_dim: int = 16, hidden_dims: Tuple[int, ...] = (512, 512, 256)):
        super(DNN2048Network, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 4))  # 4 actions
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """신경망 가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # Input shape: (batch_size, 16)
        return self.network(x)

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN 아키텍처 - 더 안정적인 학습"""
    
    def __init__(self, base_network: nn.Module):
        super(DuelingDQNNetwork, self).__init__()
        
        self.base_network = base_network
        
        # Dueling 아키텍처를 위해 base network의 마지막 레이어 제거
        if hasattr(base_network, 'fc3'):
            # CNN 네트워크의 경우
            hidden_dim = base_network.fc2.out_features
            base_network.fc3 = nn.Identity()
        else:
            # DNN 네트워크의 경우
            layers = list(base_network.network.children())
            hidden_dim = layers[-1].in_features
            base_network.network = nn.Sequential(*layers[:-1])
        
        # Value stream
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Advantage stream
        self.advantage_head = nn.Linear(hidden_dim, 4)
        
        # Initialize new heads
        nn.init.kaiming_normal_(self.value_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.value_head.bias, 0)
        nn.init.kaiming_normal_(self.advantage_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.advantage_head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        # Base features
        features = self.base_network(x)
        
        # Value and advantage
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Dueling 공식: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

def create_network(observation_type: str, use_dueling: bool = True, **kwargs) -> nn.Module:
    """네트워크 팩토리 함수"""
    
    if observation_type == 'layered':
        base_net = CNN2048Network(**kwargs)
    elif observation_type == 'flat':
        base_net = DNN2048Network(**kwargs)
    else:
        raise ValueError(f"Unknown observation_type: {observation_type}")
    
    if use_dueling:
        return DuelingDQNNetwork(base_net)
    else:
        return base_net

def count_parameters(model: nn.Module) -> int:
    """모델의 총 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 네트워크 테스트 함수
def test_networks():
    """네트워크 아키텍처 테스트"""
    print("🧠 Neural Networks 테스트")
    
    try:
        # CNN 네트워크 테스트 (Layered 관찰)
        print("\n1. CNN Network (Layered observation):")
        cnn_net = create_network('layered', use_dueling=True)
        print(f"   - 파라미터 수: {count_parameters(cnn_net):,}")
        
        # 테스트 입력
        layered_input = torch.randn(32, 4, 4, 16)  # batch_size=32
        print(f"   - 입력 shape: {layered_input.shape}")
        
        cnn_output = cnn_net(layered_input)
        print(f"   - 출력 shape: {cnn_output.shape}")
        print(f"   - 출력 범위: [{cnn_output.min():.3f}, {cnn_output.max():.3f}]")
        
        # DNN 네트워크 테스트 (Flat 관찰)
        print("\n2. DNN Network (Flat observation):")
        dnn_net = create_network('flat', use_dueling=True)
        print(f"   - 파라미터 수: {count_parameters(dnn_net):,}")
        
        # 테스트 입력
        flat_input = torch.randn(32, 16)  # batch_size=32
        print(f"   - 입력 shape: {flat_input.shape}")
        
        dnn_output = dnn_net(flat_input)
        print(f"   - 출력 shape: {dnn_output.shape}")
        print(f"   - 출력 범위: [{dnn_output.min():.3f}, {dnn_output.max():.3f}]")
        
        # 비교 테스트 - Dueling vs Non-Dueling
        print("\n3. Dueling vs Non-Dueling 비교:")
        non_dueling_net = create_network('flat', use_dueling=False)
        print(f"   - Non-Dueling 파라미터: {count_parameters(non_dueling_net):,}")
        print(f"   - Dueling 파라미터: {count_parameters(dnn_net):,}")
        
        print("\n✅ 모든 네트워크가 올바르게 작동합니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_networks()