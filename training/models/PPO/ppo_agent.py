import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time

from .networks import CNN2048ActorCritic, SimpleCNN2048ActorCritic, create_actor_critic_network
from .rollout_buffer import RolloutBuffer, EpisodeBuffer


class PPOAgent:
    """
    액션 마스킹을 지원하는 PPO (Proximal Policy Optimization) 에이전트
    
    Features:
    - Actor-Critic architecture with shared CNN backbone
    - Action masking for valid moves only
    - GAE (Generalized Advantage Estimation)
    - Clipped policy and value loss
    - Entropy regularization
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, ...] = (4, 4, 16),
        num_actions: int = 4,
        network_type: str = "cnn",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        buffer_size: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        target_kl: Optional[float] = None,
        device: str = "auto"
    ):
        """
        Args:
            state_shape: 상태 공간 형태 (4, 4, 16)
            num_actions: 액션 개수 (4)
            network_type: 네트워크 타입 ("cnn", "simple_cnn")
            lr: 학습률
            gamma: 할인 팩터
            gae_lambda: GAE 람다 파라미터
            clip_range: PPO 클리핑 범위
            value_coef: 가치 손실 계수
            entropy_coef: 엔트로피 정규화 계수
            max_grad_norm: 그래디언트 클리핑 최대값
            buffer_size: 롤아웃 버퍼 크기
            batch_size: 미니배치 크기
            n_epochs: 업데이트 에포크 수
            target_kl: 조기 종료를 위한 KL divergence 임계값
            device: 디바이스 ("auto", "cpu", "cuda")
        """
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl
        
        # 네트워크 생성
        self.actor_critic = create_actor_critic_network(
            network_type=network_type,
            input_channels=state_shape[-1],
            num_actions=num_actions
        ).to(self.device)
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        
        # 롤아웃 버퍼
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_shape=state_shape,
            num_actions=num_actions,
            device=self.device,
            gae_lambda=gae_lambda,
            gamma=gamma
        )
        
        # 에피소드 임시 버퍼
        self.episode_buffer = EpisodeBuffer(device=self.device)
        
        # 학습 통계
        self.episode_rewards = deque(maxlen=100)
        self.episode_scores = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'explained_variance': []
        }
        
        # 학습 상태
        self.steps_done = 0
        self.episodes_done = 0
        
        print(f"PPO Agent initialized on {self.device}")
        print(f"Network type: {network_type}")
        print(f"Total parameters: {sum(p.numel() for p in self.actor_critic.parameters()):,}")
    
    def select_action(
        self, 
        state: np.ndarray, 
        valid_actions: List[int],
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        액션 선택 (액션 마스킹 적용)
        
        Args:
            state: 현재 상태
            valid_actions: 유효한 액션 리스트
            training: 학습 모드 여부
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            # 상태를 텐서로 변환
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 액션 마스크 생성
            action_mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)
            action_mask[valid_actions] = True
            action_mask = action_mask.unsqueeze(0)  # 배치 차원 추가
            
            # 네트워크 forward
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(
                state_tensor, action_mask
            )
            
            return action.item(), log_prob.item(), value.item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        valid_actions: List[int]
    ):
        """경험을 에피소드 버퍼에 저장"""
        # 액션 마스크 생성
        action_mask = np.zeros(self.num_actions, dtype=bool)
        action_mask[valid_actions] = True
        
        self.episode_buffer.add(
            state=state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
            action_mask=action_mask
        )
    
    def finish_episode(self, last_value: float = 0.0):
        """
        에피소드 종료 처리
        
        Args:
            last_value: 마지막 상태의 가치 (에피소드가 끝나지 않은 경우)
        """
        if len(self.episode_buffer) > 0:
            # 에피소드 통계 업데이트
            episode_reward = sum(self.episode_buffer.rewards)
            episode_length = len(self.episode_buffer)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episodes_done += 1
            
            # 롤아웃 버퍼로 전송
            self.episode_buffer.transfer_to_rollout_buffer(self.rollout_buffer)
            self.episode_buffer.reset()
            
            # 롤아웃 버퍼가 가득 찬 경우 학습 수행
            if self.rollout_buffer.ptr == 0 and self.rollout_buffer.full:
                self.train_step(last_value)
                self.rollout_buffer.reset()
    
    def train_step(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        PPO 학습 스텝
        
        Args:
            last_value: 마지막 상태의 가치
        
        Returns:
            학습 통계 딕셔너리
        """
        # 어드밴티지와 리턴 계산
        self.rollout_buffer.compute_returns_and_advantages(last_value)
        
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0
        }
        
        n_updates = 0
        
        # 여러 에포크 동안 학습
        for epoch in range(self.n_epochs):
            # 미니배치로 학습
            for batch in self.rollout_buffer.get(self.batch_size):
                # 현재 정책으로 다시 평가
                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                    batch.states, batch.action_masks, batch.actions
                )
                
                # 정책 손실 계산 (PPO clipped loss)
                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 가치 손실 계산 (clipped loss)
                value_pred_clipped = batch.old_values + torch.clamp(
                    new_values - batch.old_values, -self.clip_range, self.clip_range
                )
                value_loss_clipped = F.mse_loss(value_pred_clipped, batch.returns)
                value_loss_unclipped = F.mse_loss(new_values, batch.returns)
                value_loss = torch.max(value_loss_clipped, value_loss_unclipped)
                
                # 엔트로피 손실
                entropy_loss = -entropy.mean()
                
                # 총 손실
                total_loss = (
                    policy_loss + 
                    self.value_coef * value_loss + 
                    self.entropy_coef * entropy_loss
                )
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 통계 업데이트
                with torch.no_grad():
                    kl_div = (batch.old_log_probs - new_log_probs).mean()
                    explained_var = 1 - F.mse_loss(new_values, batch.returns) / batch.returns.var()
                
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy_loss'] += entropy_loss.item()
                stats['total_loss'] += total_loss.item()
                stats['kl_divergence'] += kl_div.item()
                stats['explained_variance'] += explained_var.item()
                
                n_updates += 1
                
                # KL divergence 조기 종료
                if self.target_kl and kl_div > self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL divergence: {kl_div:.4f}")
                    break
            
            if self.target_kl and kl_div > self.target_kl:
                break
        
        # 평균 계산
        if n_updates > 0:
            for key in stats:
                stats[key] /= n_updates
        
        # 학습 통계 저장
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        self.steps_done += 1
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        stats = {}
        
        if self.episode_rewards:
            stats['avg_episode_reward'] = np.mean(self.episode_rewards)
            stats['avg_episode_length'] = np.mean(self.episode_lengths)
            stats['episodes_done'] = self.episodes_done
        
        if self.episode_scores:
            stats['avg_episode_score'] = np.mean(self.episode_scores)
        
        if self.training_stats['total_loss']:
            stats['avg_total_loss'] = np.mean(self.training_stats['total_loss'][-10:])
            stats['avg_policy_loss'] = np.mean(self.training_stats['policy_loss'][-10:])
            stats['avg_value_loss'] = np.mean(self.training_stats['value_loss'][-10:])
            stats['avg_kl_divergence'] = np.mean(self.training_stats['kl_divergence'][-10:])
        
        stats['steps_done'] = self.steps_done
        
        return stats
    
    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'hyperparameters': {
                'lr': self.lr,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_range': self.clip_range,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'target_kl': self.target_kl
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        
        print(f"Model loaded from {filepath}")
        print(f"Steps done: {self.steps_done}, Episodes done: {self.episodes_done}")
    
    def export_to_onnx(self, filepath: str, input_shape: Tuple[int, ...] = (4, 4, 16)):
        """ONNX 형식으로 모델 내보내기 (Actor만)"""
        self.actor_critic.eval()
        
        # 더미 입력 생성
        dummy_state = torch.randn(1, *input_shape).to(self.device)
        dummy_action_mask = torch.ones(1, self.num_actions, dtype=torch.bool).to(self.device)
        
        # ONNX 내보내기용 래퍼 클래스
        class ActorONNX(nn.Module):
            def __init__(self, actor_critic):
                super().__init__()
                self.actor_critic = actor_critic
            
            def forward(self, state, action_mask):
                logits, _ = self.actor_critic.forward(state, action_mask)
                return F.softmax(logits, dim=-1)
        
        onnx_model = ActorONNX(self.actor_critic)
        
        try:
            torch.onnx.export(
                onnx_model,
                (dummy_state, dummy_action_mask),
                filepath,
                input_names=['state', 'action_mask'],
                output_names=['action_probs'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'action_mask': {0: 'batch_size'},
                    'action_probs': {0: 'batch_size'}
                },
                opset_version=11
            )
            print(f"Model exported to ONNX: {filepath}")
        except Exception as e:
            print(f"Failed to export ONNX: {e}")
        finally:
            self.actor_critic.train()


if __name__ == "__main__":
    # 테스트 코드
    agent = PPOAgent(
        network_type="simple_cnn",
        buffer_size=128,
        batch_size=32,
        n_epochs=4
    )
    
    print("PPO Agent created successfully!")
    print(f"Device: {agent.device}")
    
    # 더미 상태로 액션 선택 테스트
    dummy_state = np.random.randn(4, 4, 16)
    valid_actions = [0, 1, 3]  # 2번 액션은 불가능
    
    action, log_prob, value = agent.select_action(dummy_state, valid_actions)
    print(f"Selected action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
    
    # 경험 저장 테스트
    agent.store_experience(
        state=dummy_state,
        action=action,
        reward=1.0,
        value=value,
        log_prob=log_prob,
        done=False,
        valid_actions=valid_actions
    )
    
    print("Experience stored successfully!")
    print(f"Episode buffer length: {len(agent.episode_buffer)}")
