import numpy as np
import torch
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional

# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """DQN용 경험 재생 버퍼"""
    
    def __init__(self, capacity: int, seed: Optional[int] = None):
        """
        Args:
            capacity: 버퍼 최대 크기
            seed: 랜덤 시드
        """
        self.capacity = capacity  # 🔥 추가: capacity 속성
        self.buffer = []
        self.position = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """경험을 버퍼에 추가"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, ...]:
        """배치 샘플링 - 성능 최적화 버전"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"버퍼 크기({len(self.buffer)})가 배치 크기({batch_size})보다 작습니다.")
        
        # 랜덤 샘플링
        experiences = random.sample(self.buffer, batch_size)
        
        # 효율적인 텐서 변환 - 경고 해결
        states_list = [e.state for e in experiences]
        actions_list = [e.action for e in experiences]
        rewards_list = [e.reward for e in experiences]
        next_states_list = [e.next_state for e in experiences]
        dones_list = [e.done for e in experiences]
        
        # NumPy 배열로 먼저 변환 후 텐서로 변환 (성능 개선)
        states = torch.from_numpy(np.array(states_list, dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array(actions_list, dtype=np.int64)).to(device)
        rewards = torch.from_numpy(np.array(rewards_list, dtype=np.float32)).to(device)
        next_states = torch.from_numpy(np.array(next_states_list, dtype=np.float32)).to(device)
        # bool 경고 해결: 명시적으로 bool로 변환
        dones = torch.from_numpy(np.array(dones_list, dtype=bool)).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """버퍼 크기 반환"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """배치 크기만큼 샘플이 준비되었는지 확인"""
        return len(self) >= batch_size

class PrioritizedReplayBuffer:
    """우선순위 경험 재생 버퍼"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_frames: int = 100000, seed: Optional[int] = None):
        """
        Args:
            capacity: 버퍼 최대 크기
            alpha: 우선순위 지수 (0=uniform, 1=full priority)
            beta: importance sampling 시작값
            beta_frames: beta가 1에 도달하는 프레임 수
            seed: 랜덤 시드
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta
        self.beta = beta
        self.beta_frames = beta_frames
        self.frame = 1
        
        # 🔥 누락된 속성들 추가
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        
        self.tree = np.zeros(2 * self.tree_capacity - 1)
        self.data = np.empty(self.tree_capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0  # 🔥 이 속성이 누락되었음!
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _update_tree(self, tree_index: int, priority: float):
        """트리 업데이트"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # 부모 노드들 업데이트
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
        """경험을 버퍼에 추가"""
        experience = Experience(state, action, reward, next_state, done)
        
        # 우선순위 계산 (TD error 기반)
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
        """우선순위 기반 배치 샘플링"""
        if self.size < batch_size:
            raise ValueError(f"버퍼 크기({self.size})가 배치 크기({batch_size})보다 작습니다.")
        
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
        sampling_probabilities = np.array(priorities, dtype=np.float64) / self.tree[0]
        is_weights = np.power(self.size * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        
        # 경험 추출
        experiences = [self.data[i] for i in indices]
        
        # 효율적인 텐서 변환
        states_list = [e.state for e in experiences]
        actions_list = [e.action for e in experiences]
        rewards_list = [e.reward for e in experiences]
        next_states_list = [e.next_state for e in experiences]
        dones_list = [e.done for e in experiences]
        
        states = torch.from_numpy(np.array(states_list, dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array(actions_list, dtype=np.int64)).to(device)
        rewards = torch.from_numpy(np.array(rewards_list, dtype=np.float32)).to(device)
        next_states = torch.from_numpy(np.array(next_states_list, dtype=np.float32)).to(device)
        dones = torch.from_numpy(np.array(dones_list, dtype=bool)).to(device)
        is_weights = torch.from_numpy(np.array(is_weights, dtype=np.float32)).to(device)
        
        return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: List[int], errors: List[float]):
        """TD error를 기반으로 우선순위 업데이트"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            tree_index = idx + self.tree_capacity - 1
            self._update_tree(tree_index, priority)
    
    def __len__(self) -> int:
        """버퍼 크기 반환"""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """배치 크기만큼 샘플이 준비되었는지 확인"""
        return self.size >= batch_size

# 테스트 함수
def test_replay_buffer():
    """Replay Buffer 테스트"""
    print("💾 Replay Buffer 테스트")
    
    # 기본 Replay Buffer 테스트
    print("\n1. 기본 Replay Buffer:")
    buffer = ReplayBuffer(capacity=1000)
    
    # 더미 데이터 추가
    for i in range(100):
        state = np.random.randn(16).astype(np.float32)  # 명시적 타입 지정
        action = int(np.random.randint(4))  # 명시적 int 변환
        reward = float(np.random.randn())  # 명시적 float 변환
        next_state = np.random.randn(16).astype(np.float32)
        done = bool(np.random.choice([True, False]))  # 명시적 bool 변환
        buffer.push(state, action, reward, next_state, done)
    
    print(f"   - 버퍼 크기: {len(buffer)}")
    print(f"   - 샘플링 준비: {buffer.is_ready(32)}")
    
    # 샘플링 테스트
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"   - 샘플 shapes: {states.shape}, {actions.shape}, {rewards.shape}")
    print(f"   - 데이터 타입: {states.dtype}, {actions.dtype}, {dones.dtype}")
    
    # 우선순위 Replay Buffer 테스트
    print("\n2. 우선순위 Replay Buffer:")
    priority_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # 더미 데이터 추가
    for i in range(100):
        state = np.random.randn(16).astype(np.float32)
        action = int(np.random.randint(4))
        reward = float(np.random.randn())
        next_state = np.random.randn(16).astype(np.float32)
        done = bool(np.random.choice([True, False]))
        error = abs(float(np.random.randn()))  # TD error
        priority_buffer.push(state, action, reward, next_state, done, error)
    
    print(f"   - 버퍼 크기: {len(priority_buffer)}")
    print(f"   - 샘플링 준비: {priority_buffer.is_ready(32)}")
    
    # 우선순위 샘플링 테스트
    result = priority_buffer.sample(32)
    states, actions, rewards, next_states, dones, is_weights, indices = result
    print(f"   - 샘플 shapes: {states.shape}, {actions.shape}, {is_weights.shape}")
    print(f"   - IS weights 범위: [{is_weights.min():.3f}, {is_weights.max():.3f}]")
    print(f"   - 데이터 타입: {states.dtype}, {actions.dtype}, {dones.dtype}")
    
    print("\n 모든 Replay Buffer가 올바르게 작동합니다!")

if __name__ == "__main__":
    test_replay_buffer()