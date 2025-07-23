import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class CNN2048Network(nn.Module):
    """2048용 CNN 네트워크 - 개선된 버전"""
    
    def __init__(self, input_channels: int = 16, hidden_dim: int = 512):
        super(CNN2048Network, self).__init__()
        
        # Convolutional layers - 기존 유지하되 BatchNorm 추가
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # BatchNorm 추가 (DDQN 코드의 장점)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 기존 FC 레이어 구조 유지
        conv_output_size = 4 * 4 * 128
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)
        
        # Dropout 추가 (안정성 향상)
        self.dropout = nn.Dropout(0.1)
        
        # 개선된 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming 초기화 적용"""
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
        """순전파 - BatchNorm 적용"""
        # 차원 변환 (기존 로직 유지)
        if x.dim() == 4 and x.shape[-1] == 16:
            x = x.permute(0, 3, 1, 2)
        
        # Conv layers with BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # FC layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 기존 함수들 유지하되 Dueling 옵션 추가
class DuelingDQNNetwork(nn.Module):
    """Dueling DQN - 선택적 적용"""
    
    def __init__(self, base_network: nn.Module):
        super(DuelingDQNNetwork, self).__init__()
        
        self.base_network = base_network
        
        # 마지막 레이어 제거하고 Dueling head 추가
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
        
        # Dueling 공식
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

def create_network(use_dueling: bool = False, **kwargs) -> nn.Module:
    """네트워크 팩토리 함수 - 기존 호환성 유지"""
    base_net = CNN2048Network(**kwargs)
    
    if use_dueling:
        return DuelingDQNNetwork(base_net)
    else:
        return base_net

# 기존 함수들 유지
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 네트워크 테스트 함수
def test_networks():
    """네트워크 아키텍처 테스트"""
    print("🧠 Neural Networks 테스트")
    
    try:
        # CNN 네트워크 테스트 (Layered 관찰)
        print("\n1. CNN Network (Layered observation):")
        cnn_net_dueling = create_network(use_dueling=True)
        cnn_net_simple = create_network(use_dueling=False)

        print(f"   - Dueling 파라미터 수: {count_parameters(cnn_net_dueling):,}")
        print(f"   - Non-Dueling 파라미터 수: {count_parameters(cnn_net_simple):,}")
        
        # 테스트 입력
        layered_input = torch.randn(32, 4, 4, 16)  # batch_size=32
        print(f"   - 입력 shape: {layered_input.shape}")
        
        cnn_output = cnn_net_dueling(layered_input)
        print(f"   - 출력 shape: {cnn_output.shape}")
        print(f"   - 출력 범위: [{cnn_output.min():.3f}, {cnn_output.max():.3f}]")
        
        print("\n✅ 모든 네트워크가 올바르게 작동합니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_networks()
