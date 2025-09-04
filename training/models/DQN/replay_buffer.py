import numpy as np
import torch
import random
from collections import namedtuple
from typing import List, Tuple, Optional

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

def _experiences_to_tensors(experiences: List[Experience], device: torch.device) -> Tuple[torch.Tensor, ...]:
    """경험을 텐서로 변환하는 공통 함수"""
    states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
    actions = torch.LongTensor([e.action for e in experiences]).to(device)
    rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
    next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
    dones = torch.BoolTensor([e.done for e in experiences]).to(device)
    return states, actions, rewards, next_states, dones

class ReplayBuffer:
    """기본 경험 재생 버퍼"""
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """경험 저장"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, ...]:
        """배치 샘플링"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size ({len(self.buffer)}) < batch size ({batch_size})")
        
        experiences = random.sample(self.buffer, batch_size)
        return _experiences_to_tensors(experiences, device)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size

class PrioritizedReplayBuffer:
    """우선순위 경험 재생 버퍼"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, seed: Optional[int] = None):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta
        self.beta_frames = 100000  # 고정값으로 단순화
        self.frame = 1
        
        # Sum tree 초기화
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        
        self.tree = np.zeros(2 * self.tree_capacity - 1)
        self.data = np.empty(self.tree_capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _update_tree(self, tree_index: int, priority: float):
        """트리 업데이트"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def _get_leaf(self, value: float) -> Tuple[int, float, int]:
        """값에 해당하는 리프 노드 찾기"""
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.tree_capacity + 1
        return leaf_index, self.tree[leaf_index], data_index
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, error: Optional[float] = None) -> None:
        """경험 저장"""
        experience = Experience(state, action, reward, next_state, done)
        
        # 우선순위 계산
        if error is None:
            priority = np.max(self.tree[-self.tree_capacity:]) if self.size > 0 else 1.0
        else:
            priority = (abs(error) + 1e-6) ** self.alpha
        
        # 데이터 저장
        self.data[self.data_pointer] = experience
        tree_index = self.data_pointer + self.tree_capacity - 1
        self._update_tree(tree_index, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.tree_capacity
        if self.size < self.tree_capacity:
            self.size += 1
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')):
        """우선순위 기반 샘플링"""
        if self.size < batch_size:
            raise ValueError(f"Buffer size ({self.size}) < batch size ({batch_size})")
        
        # Beta 스케줄링
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        # 샘플링
        indices = []
        priorities = []
        segment = self.tree[0] / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            index, priority, data_index = self._get_leaf(value)
            indices.append(data_index)
            priorities.append(priority)
        
        # Importance sampling weights
        total_priority = self.tree[0]
        if total_priority <= 0:
            # 우선순위 합이 0이면 균등 샘플링
            is_weights = np.ones(batch_size, dtype=np.float32)
        else:
            sampling_probabilities = np.array(priorities, dtype=np.float64) / total_priority
            is_weights = np.power(self.size * sampling_probabilities, -beta)
            is_weights = is_weights / is_weights.max()  # 정규화
            is_weights = is_weights.astype(np.float32)
        
        # 경험 추출 및 텐서 변환
        experiences = [self.data[i] for i in indices]
        states, actions, rewards, next_states, dones = _experiences_to_tensors(experiences, device)
        is_weights = torch.FloatTensor(is_weights).to(device)
        
        return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: List[int], errors: List[float]):
        """우선순위 업데이트"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            tree_index = idx + self.tree_capacity - 1
            self._update_tree(tree_index, priority)
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size