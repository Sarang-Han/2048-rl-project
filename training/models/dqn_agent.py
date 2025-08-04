import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple, List
import os

from models.networks import create_network, count_parameters
from models.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    """2048용 DQN 에이전트"""
    
    def __init__(self, 
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 50000,
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 target_update: int = 1000,
                 double_dqn: bool = True,
                 dueling: bool = False,
                 prioritized_replay: bool = True,
                 device: Optional[str] = None,
                 seed: Optional[int] = None):
        
        # 시드 설정
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 하이퍼파라미터
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.prioritized_replay = prioritized_replay
        
        # 네트워크 생성
        self.q_network = create_network(use_dueling=dueling).to(self.device)
        self.target_network = create_network(use_dueling=dueling).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 경험 재생 버퍼
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, seed=seed)
        else:
            self.memory = ReplayBuffer(buffer_size, seed=seed)
        
        # 학습 상태
        self.steps_done = 0
        self.episode_rewards = []
    
    def get_epsilon(self) -> float:
        """현재 탐험률 반환"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon
    
    def select_action(self, state: np.ndarray, training: bool = True, 
                     valid_actions: Optional[List[int]] = None) -> int:
        """액션 선택 (액션 마스킹 지원)"""
        
        # 유효한 액션이 제공되지 않은 경우 모든 액션 허용
        if valid_actions is None:
            valid_actions = [0, 1, 2, 3]
        
        # 탐험
        if training and random.random() < self.get_epsilon():
            return random.choice(valid_actions)
        
        # 착취
        was_training = self.q_network.training
        self.q_network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)[0]
            
            # 액션 마스킹
            masked_q_values = q_values.clone()
            for i in range(4):
                if i not in valid_actions:
                    masked_q_values[i] = float('-inf')
            
            action = masked_q_values.argmax().item()
        
        if was_training:
            self.q_network.train()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """경험 저장"""
        if self.prioritized_replay and td_error is not None:
            self.memory.push(state, action, reward, next_state, done, td_error)
        else:
            self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """학습 스텝"""
        if not self.memory.is_ready(self.batch_size):
            return None
        
        self.q_network.train()
        
        # 샘플링
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, is_weights, indices = \
                self.memory.sample(self.batch_size, self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size, self.device)
            is_weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # Q값 계산
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 타겟 Q값 계산
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones.unsqueeze(1)))
        
        # 손실 계산
        td_errors = target_q_values - current_q_values
        
        if self.prioritized_replay:
            loss = (td_errors.pow(2) * is_weights.unsqueeze(1)).mean()
            if indices is not None:
                td_errors_np = td_errors.detach().cpu().numpy().flatten()
                self.memory.update_priorities(indices, td_errors_np)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 타겟 네트워크 업데이트
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        return loss.item()
    
    def get_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {
            'steps_done': self.steps_done,
            'epsilon': self.get_epsilon(),
            'memory_size': len(self.memory),
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        }
        return stats
    
    def save_model(self, filepath: str):
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update': self.target_update,
                'double_dqn': self.double_dqn,
                'dueling': self.dueling,
                'prioritized_replay': self.prioritized_replay
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
    
    def export_to_onnx(self, filepath: str, input_shape: Tuple[int, ...] = (4, 4, 16)):
        """ONNX 모델 내보내기"""
        self.q_network.eval()
        
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        
        torch.onnx.export(
            self.q_network,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['q_values']
        )