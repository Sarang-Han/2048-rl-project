import torch
import numpy as np
from typing import Generator, Optional, Tuple, NamedTuple
from collections import deque

class RolloutBufferSample(NamedTuple):
    """PPO rollout buffer sample"""
    states: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor

class RolloutBuffer:
    """
    PPO용 롤아웃 버퍼
    
    GAE(Generalized Advantage Estimation)와 리턴 계산을 지원합니다.
    """
    
    def __init__(
        self, 
        buffer_size: int,
        state_shape: Tuple[int, ...],
        num_actions: int,
        device: torch.device = torch.device("cpu"),
        gae_lambda: float = 0.95,
        gamma: float = 0.99
    ):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        self.ptr = 0
        self.full = False
        
        # 버퍼 초기화
        self.states = torch.zeros((buffer_size,) + state_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.action_masks = torch.zeros((buffer_size, num_actions), dtype=torch.bool, device=device)
        
        # 계산된 값들
        self.advantages = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: np.ndarray
    ):
        """버퍼에 경험 추가"""
        self.states[self.ptr] = torch.from_numpy(state).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = torch.from_numpy(action_mask).to(self.device)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """
        GAE를 사용하여 어드밴티지와 리턴 계산
        
        Args:
            last_value: 마지막 상태의 가치 (에피소드가 끝나지 않은 경우)
        """
        last_gae_lam = 0
        
        # 실제 사용된 버퍼 크기
        buffer_size = self.buffer_size if self.full else self.ptr
        
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = self.values[step + 1]
            
            # TD error 계산
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE 계산
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # 리턴 = 어드밴티지 + 가치
        self.returns = self.advantages + self.values
    
    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSample, None, None]:
        """
        미니배치 생성기
        
        Args:
            batch_size: 배치 크기 (None이면 전체 버퍼 반환)
        """
        # 어드밴티지 정규화
        buffer_size = self.buffer_size if self.full else self.ptr
        advantages = self.advantages[:buffer_size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if batch_size is None:
            # 전체 버퍼 반환
            yield RolloutBufferSample(
                states=self.states[:buffer_size],
                actions=self.actions[:buffer_size],
                old_values=self.values[:buffer_size],
                old_log_probs=self.log_probs[:buffer_size],
                advantages=advantages,
                returns=self.returns[:buffer_size],
                action_masks=self.action_masks[:buffer_size]
            )
        else:
            # 미니배치로 분할
            indices = torch.randperm(buffer_size, device=self.device)
            
            for start_idx in range(0, buffer_size, batch_size):
                end_idx = min(start_idx + batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                yield RolloutBufferSample(
                    states=self.states[batch_indices],
                    actions=self.actions[batch_indices],
                    old_values=self.values[batch_indices],
                    old_log_probs=self.log_probs[batch_indices],
                    advantages=advantages[batch_indices],
                    returns=self.returns[batch_indices],
                    action_masks=self.action_masks[batch_indices]
                )
    
    def reset(self):
        """버퍼 초기화"""
        self.ptr = 0
        self.full = False
    
    def size(self) -> int:
        """현재 버퍼 크기 반환"""
        return self.buffer_size if self.full else self.ptr


class EpisodeBuffer:
    """
    에피소드별 임시 버퍼 (롤아웃 수집용)
    """
    
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.reset()
    
    def reset(self):
        """버퍼 초기화"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: np.ndarray
    ):
        """경험 추가"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
    
    def transfer_to_rollout_buffer(self, rollout_buffer: RolloutBuffer):
        """에피소드 데이터를 롤아웃 버퍼로 전송"""
        for i in range(len(self.states)):
            rollout_buffer.add(
                state=self.states[i],
                action=self.actions[i],
                reward=self.rewards[i],
                value=self.values[i],
                log_prob=self.log_probs[i],
                done=self.dones[i],
                action_mask=self.action_masks[i]
            )
    
    def __len__(self):
        return len(self.states)


if __name__ == "__main__":
    # 테스트 코드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 롤아웃 버퍼 테스트
    buffer = RolloutBuffer(
        buffer_size=100,
        state_shape=(4, 4, 16),
        num_actions=4,
        device=device
    )
    
    print(f"Buffer created with size: {buffer.buffer_size}")
    
    # 샘플 데이터 추가
    for i in range(50):
        state = np.random.randn(4, 4, 16)
        action = np.random.randint(0, 4)
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        done = np.random.choice([True, False], p=[0.1, 0.9])
        action_mask = np.random.choice([True, False], size=4, p=[0.8, 0.2])
        
        buffer.add(state, action, reward, value, log_prob, done, action_mask)
    
    print(f"Added {buffer.size()} experiences")
    
    # 어드밴티지 계산
    buffer.compute_returns_and_advantages(last_value=0.0)
    
    # 배치 생성 테스트
    for batch in buffer.get(batch_size=16):
        print(f"Batch states shape: {batch.states.shape}")
        print(f"Batch actions shape: {batch.actions.shape}")
        print(f"Batch advantages mean: {batch.advantages.mean():.4f}")
        break
