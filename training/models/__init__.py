"""
2048 DQN 모델 패키지

DQN 에이전트와 관련된 모든 구성요소를 포함합니다.
- networks: CNN, DNN 신경망 아키텍처
- dqn_agent: DQN 에이전트 (Double DQN, Dueling DQN 지원)
- replay_buffer: 경험 재생 버퍼 (일반 및 우선순위)
"""

# 핵심 클래스들 import - 실제 존재하는 클래스만 import
from .networks import (
    CNN2048Network, 
    DNN2048Network, 
    create_network, 
    count_parameters
)

from .replay_buffer import (
    ReplayBuffer, 
    PrioritizedReplayBuffer,
    Experience
)

from .dqn_agent import DQNAgent

# 패키지에서 공개할 클래스/함수들 - DuelingDQN 제거
__all__ = [
    # Networks
    'CNN2048Network',
    'DNN2048Network', 
    'create_network',
    'count_parameters',
    
    # Replay Buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'Experience',
    
    # Agent
    'DQNAgent'
]

# 기본 모델 설정
DEFAULT_MODEL_CONFIG = {
    'observation_type': 'flat',
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 50000,
    'buffer_size': 100000,
    'batch_size': 32,
    'target_update': 1000,
    'double_dqn': True,
    'dueling': True,
    'prioritized_replay': True
}

def create_agent(observation_type='flat', **kwargs):
    """
    편의 함수: DQN 에이전트 생성
    
    Args:
        observation_type: 'flat' 또는 'layered'
        **kwargs: 추가 에이전트 설정
    
    Returns:
        DQNAgent: 초기화된 DQN 에이전트
    """
    config = DEFAULT_MODEL_CONFIG.copy()
    config.update(kwargs)
    config['observation_type'] = observation_type
    return DQNAgent(**config)

# 지원되는 네트워크 타입
SUPPORTED_NETWORKS = {
    'flat': 'DNN2048Network',
    'layered': 'CNN2048Network'
}

print(f"✅ Models 패키지 로드 완료 - 지원 네트워크: {list(SUPPORTED_NETWORKS.keys())}")