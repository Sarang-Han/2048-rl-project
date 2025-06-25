"""
2048 DQN 학습 패키지

학습 스크립트, 설정, 유틸리티를 포함합니다.
- 학습 설정 관리
- 학습 모니터링 및 시각화
- 모델 저장/로드 유틸리티
"""

import os
import sys

# 학습 기본 설정
DEFAULT_TRAINING_CONFIG = {
    # 기본 설정
    'episodes': 2000,
    'max_steps_per_episode': 1000,
    'device': 'auto',  # 'auto', 'cpu', 'cuda'
    
    # DQN 하이퍼파라미터
    'buffer_size': 100000,
    'batch_size': 64,
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 50000,
    'target_update': 1000,
    
    # 평가 및 저장 설정
    'eval_interval': 100,
    'eval_episodes': 10,
    'save_interval': 500,
    'plot_interval': 50,
    'log_interval': 10,
    
    # 고급 기능
    'double_dqn': True,
    'dueling': True,
    'prioritized_replay': True,
    
    # 저장 경로
    'save_dir': 'models',
    'log_dir': 'logs'
}

def get_training_config(custom_config=None):
    """
    학습 설정 반환
    
    Args:
        custom_config: 사용자 정의 설정 딕셔너리
    
    Returns:
        dict: 병합된 학습 설정
    """
    config = DEFAULT_TRAINING_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    
    # device 자동 설정
    if config['device'] == 'auto':
        import torch
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

def setup_directories(base_path='/content/drive/MyDrive/2048_models'):
    """
    학습에 필요한 디렉토리 생성
    
    Args:
        base_path: 기본 저장 경로
    """
    directories = [
        base_path,
        os.path.join(base_path, 'checkpoints'),
        os.path.join(base_path, 'best_models'),
        os.path.join(base_path, 'onnx_models'),
        os.path.join(base_path, 'logs'),
        os.path.join(base_path, 'plots')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

# 패키지 정보
__all__ = [
    'DEFAULT_TRAINING_CONFIG',
    'get_training_config',
    'setup_directories',
    'is_colab_environment',
    'mount_drive'
]