"""
2048 CNN-DQN 모델 패키지

CNN 기반 DQN 에이전트와 관련된 모든 구성요소를 포함합니다.
- networks: CNN 신경망 아키텍처 (layered 관찰 전용)
- dqn_agent: DQN 에이전트 (Double DQN, Dueling DQN 지원)
- replay_buffer: 경험 재생 버퍼 (일반 및 우선순위)
"""

# 핵심 클래스들 import - 실제 존재하는 클래스만 import
from .networks import (
    CNN2048Network, 
    create_network, 
    count_parameters
)

from .replay_buffer import (
    ReplayBuffer, 
    PrioritizedReplayBuffer,
    Experience
)

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
    'observation_type': 'layered',  # CNN 전용
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

def create_agent(**kwargs):
    """CNN-DQN 에이전트 생성 팩토리 함수"""
    config = DEFAULT_MODEL_CONFIG.copy()
    config.update(kwargs)
    # observation_type 강제 설정
    config['observation_type'] = 'layered'
    return DQNAgent(**config)

SUPPORTED_NETWORKS = {
    'layered': 'CNN2048Network'
}

SUPPORTED_OBSERVATION_TYPES = ['layered']

def get_model_info():
    """모델 패키지 정보 출력"""
    print("📊 CNN-DQN 모델 패키지 정보:")
    print(f"  지원 네트워크: {list(SUPPORTED_NETWORKS.keys())}")
    print(f"  지원 관찰 타입: {SUPPORTED_OBSERVATION_TYPES}")
    print(f"  기본 설정: Double DQN, Dueling DQN, Prioritized Replay")

print(f"✅ Models 패키지 로드 완료 - 지원 네트워크: {list(SUPPORTED_NETWORKS.keys())}")