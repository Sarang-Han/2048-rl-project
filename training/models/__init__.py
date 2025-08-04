"""
2048 CNN-DQN 모델 패키지

CNN 기반 DQN 에이전트와 관련된 모든 구성요소를 포함합니다.
"""

from .networks import CNN2048Network, create_network, count_parameters
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Experience
from .dqn_agent import DQNAgent

__all__ = [
    # Networks
    'CNN2048Network',
    'create_network',
    'count_parameters',
    
    # Replay Buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer', 
    'Experience',
    
    # Agent
    'DQNAgent'
]

DEFAULT_MODEL_CONFIG = {
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 50000,
    'buffer_size': 100000,
    'batch_size': 32,
    'target_update': 1000,
    'double_dqn': True,
    'dueling': False,
    'prioritized_replay': True
}

def create_agent(**kwargs):
    """CNN-DQN 에이전트 생성 팩토리 함수"""
    config = DEFAULT_MODEL_CONFIG.copy()
    config.update(kwargs)
    return DQNAgent(**config)

print("Models 패키지 로드 완료")