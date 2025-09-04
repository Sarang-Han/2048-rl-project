import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional

class CNN2048ActorCritic(nn.Module):
    """
    2048 게임을 위한 CNN 기반 Actor-Critic 네트워크
    
    Features:
    - Shared CNN backbone for both actor and critic
    - Action masking support
    - Separate heads for policy and value
    """
    
    def __init__(self, input_channels: int = 16, num_actions: int = 4):
        super(CNN2048ActorCritic, self).__init__()
        
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        # Shared CNN backbone (BatchNorm 제거 - PPO는 batch_size=1로 학습)
        self.shared_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )
        
        # Calculate flattened feature size
        # After 3 conv layers with kernel_size=2, stride=1, padding=0: 4x4 -> 3x3 -> 2x2 -> 1x1
        self.feature_size = 128 * 1 * 1
        
        # Shared dense layers
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic
        
        Args:
            x: Input state tensor [batch_size, 4, 4, 16]
            action_mask: Boolean mask for valid actions [batch_size, 4]
        
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Ensure input is in correct format: [batch_size, channels, height, width]
        if x.dim() == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        elif x.dim() == 3 and x.shape[-1] == self.input_channels:
            x = x.unsqueeze(0).permute(0, 3, 1, 2)  # [H, W, C] -> [1, C, H, W]
        
        # Shared CNN features
        conv_features = self.shared_cnn(x)  # [batch_size, 128, 1, 1]
        features = conv_features.view(conv_features.size(0), -1)  # [batch_size, 128]
        
        # Shared dense features
        shared_features = self.shared_fc(features)  # [batch_size, 256]
        
        # Actor head - policy logits
        action_logits = self.actor_head(shared_features)  # [batch_size, 4]
        
        # Apply action masking if provided
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
        
        # Critic head - state value
        state_value = self.critic_head(shared_features)  # [batch_size, 1]
        
        return action_logits, state_value.squeeze(-1)
    
    def get_action_and_value(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None, 
                           action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value for PPO training
        
        Args:
            x: Input state
            action_mask: Valid action mask
            action: Specific action to evaluate (for training)
        
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        logits, value = self.forward(x, action_mask)
        
        # Create categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        # Get action if not provided
        if action is None:
            action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the value estimate"""
        _, value = self.forward(x)
        return value


class SimpleCNN2048ActorCritic(nn.Module):
    """
    Simpler CNN-based Actor-Critic for faster training
    """
    
    def __init__(self, input_channels: int = 16, num_actions: int = 4):
        super(SimpleCNN2048ActorCritic, self).__init__()
        
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        # Simpler CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Adaptive pooling for flexible input size
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(),
        )
        
        # Actor and Critic heads
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        # Format input
        if x.dim() == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3 and x.shape[-1] == self.input_channels:
            x = x.unsqueeze(0).permute(0, 3, 1, 2)
        
        features = self.backbone(x)
        
        logits = self.actor(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        value = self.critic(features).squeeze(-1)
        
        return logits, value
    
    def get_action_and_value(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None, 
                           action: Optional[torch.Tensor] = None):
        logits, value = self.forward(x, action_mask)
        
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value
    
    def get_value(self, x: torch.Tensor):
        _, value = self.forward(x)
        return value


def create_actor_critic_network(network_type: str = "cnn", **kwargs) -> nn.Module:
    """
    Factory function to create Actor-Critic networks
    
    Args:
        network_type: Type of network ("cnn", "simple_cnn")
        **kwargs: Additional arguments for network creation
    
    Returns:
        Actor-Critic network
    """
    if network_type.lower() == "cnn":
        return CNN2048ActorCritic(**kwargs)
    elif network_type.lower() == "simple_cnn":
        return SimpleCNN2048ActorCritic(**kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test data
    batch_size = 2
    state = torch.randn(batch_size, 4, 4, 16).to(device)
    action_mask = torch.tensor([[True, True, False, True], [True, False, True, True]]).to(device)
    
    # Test CNN Actor-Critic
    print("Testing CNN2048ActorCritic...")
    model = CNN2048ActorCritic().to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    logits, value = model(state, action_mask)
    print(f"Logits shape: {logits.shape}, Value shape: {value.shape}")
    
    action, log_prob, entropy, value = model.get_action_and_value(state, action_mask)
    print(f"Action: {action}, Log_prob: {log_prob}, Entropy: {entropy}, Value: {value}")
    
    # Test Simple CNN Actor-Critic
    print("\nTesting SimpleCNN2048ActorCritic...")
    simple_model = SimpleCNN2048ActorCritic().to(device)
    print(f"Parameters: {count_parameters(simple_model):,}")
    
    logits, value = simple_model(state, action_mask)
    print(f"Logits shape: {logits.shape}, Value shape: {value.shape}")
