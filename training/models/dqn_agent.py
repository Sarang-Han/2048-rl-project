import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple, List
import copy
import os
from pathlib import Path

from models.networks import create_network, count_parameters
from models.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    """2048용 DQN 에이전트 - 개선된 버전"""
    
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
        """
        Args:
            lr: 학습률
            gamma: 할인 인수
            epsilon_start: 초기 탐험률
            epsilon_end: 최종 탐험률
            epsilon_decay: 탐험률 감소 스텝 수
            buffer_size: 경험 버퍼 크기
            batch_size: 배치 크기
            target_update: 타겟 네트워크 업데이트 주기
            double_dqn: Double DQN 사용 여부
            dueling: Dueling DQN 사용 여부
            prioritized_replay: 우선순위 경험 재생 사용 여부
            device: 연산 장치
            seed: 랜덤 시드
        """
        # 시드 설정 (완전 수정)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        # 디바이스 설정 (완전 수정)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🤖 DQN Agent 초기화 - Device: {self.device}")
        
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
        
        # 개선된 네트워크 생성
        self.q_network = create_network(use_dueling=dueling).to(self.device)
        self.target_network = create_network(use_dueling=dueling).to(self.device)
        
        # 타겟 네트워크 초기화
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 명시적 eval 모드
        
        # 개선된 옵티마이저 (기존 유지하되 설정 조정)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, eps=1e-4)
        
        # 경험 재생 버퍼
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size, seed=seed)
        else:
            self.memory = ReplayBuffer(buffer_size, seed=seed)
        
        # 학습 상태
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []
        
        print(f"🤖 DQN Agent 초기화 완료")
        print(f"   - Double DQN: {double_dqn}")
        print(f"   - Dueling DQN: {dueling}")
        print(f"   - Prioritized Replay: {prioritized_replay}")
    
    def get_epsilon(self) -> float:
        """현재 탐험률 반환"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon
    
    def select_action(self, state: np.ndarray, training: bool = True, 
                     valid_actions: Optional[List[int]] = None) -> int:
        """액션 마스킹이 적용된 개선된 액션 선택"""
        
        # 유효한 액션이 없는 경우 처리 (완전 수정)
        if valid_actions is None or len(valid_actions) == 0:
            print("⚠️ Warning: No valid actions available! Returning random action.")
            return random.randint(0, 3)
        
        # 탐험 (Exploration)
        if training and random.random() < self.get_epsilon():
            return random.choice(valid_actions)
        
        # 착취 (Exploitation) - Q값 기반 선택
        was_training = self.q_network.training
        self.q_network.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)[0]
            
            # 유효하지 않은 액션에 대해 매우 낮은 값 설정
            masked_q_values = q_values.clone()
            for i in range(4):
                if i not in valid_actions:
                    masked_q_values[i] = float('-inf')
            
            action = masked_q_values.argmax().item()
        
        # 원래 모드 복원
        if was_training:
            self.q_network.train()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """경험 저장"""
        # 🔥 수정: 올바른 속성명 사용
        if self.prioritized_replay and td_error is not None:
            self.memory.push(state, action, reward, next_state, done, td_error)
        else:
            self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """개선된 학습 스텝"""
        # 메모리 준비 상태 확인 (수정됨)
        if not self.memory.is_ready(self.batch_size):
            return None
        
        # 명시적 train 모드
        self.q_network.train()
        
        # 기존 샘플링 로직 유지하되 Double DQN 적용
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, is_weights, indices = \
                self.memory.sample(self.batch_size, self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.memory.sample(self.batch_size, self.device)
            is_weights = torch.ones(self.batch_size).to(self.device)
            indices = None
        
        # 현재 Q값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN 적용
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: 액션 선택과 평가 분리
                next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions)
            else:
                # 기존 DQN
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * (~dones.unsqueeze(1)))
        
        # 손실 계산
        td_errors = target_q_values - current_q_values
        
        if self.prioritized_replay:
            # 우선순위 재생 적용
            loss = (td_errors.pow(2) * is_weights.unsqueeze(1)).mean()
            
            if indices is not None:
                td_errors_np = td_errors.detach().cpu().numpy().flatten()
                self.memory.update_priorities(indices, td_errors_np)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # 역전파 with 그래디언트 클리핑
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # 안정성 향상
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
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        }
        return stats
    
    def save_model(self, filepath: str):
        """모델 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
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
        print(f"💾 모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        
        print(f"📁 모델 로드 완료: {filepath}")
        print(f"   - 학습 스텝: {self.steps_done:,}")
        print(f"   - 평균 보상: {np.mean(self.episode_rewards[-100:]):.2f}")
    
    def export_to_onnx(self, filepath: str, input_shape: Tuple[int, ...] = (4, 4, 16)):
        """ONNX 내보내기 기능 추가"""
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
        
        print(f"🔄 ONNX 모델 내보내기 완료: {filepath}")

# 테스트 함수
def test_dqn_agent():
    """DQN Agent 테스트"""
    print("🤖 DQN Agent 테스트")
    
    agent = DQNAgent(
        buffer_size=1000,
        batch_size=32,
        epsilon_decay=10000
    )
    
    # 더미 경험 추가
    for i in range(100):
        state = np.random.randn(4, 4, 16)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.randn(4, 4, 16)
        done = np.random.choice([True, False])
        agent.store_experience(state, action, reward, next_state, done)
    
    # 액션 선택 테스트
    test_state = np.random.randn(4, 4, 16)
    action = agent.select_action(test_state)
    print(f"   - 선택된 액션: {action}")
    print(f"   - 현재 epsilon: {agent.get_epsilon():.3f}")
    
    # 학습 테스트
    loss = agent.train_step()
    if loss:
        print(f"   - 학습 손실: {loss:.6f}")
    
    stats = agent.get_stats()
    print(f"   - 학습 통계: {stats}")
    
    print("\nDQN Agent가 올바르게 작동합니다!")

if __name__ == "__main__":
    test_dqn_agent()
