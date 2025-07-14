"""
2048 CNN-DQN 학습 패키지

CNN 기반 DQN 학습을 위한 스크립트, 설정, 유틸리티를 포함합니다.
- CNN 전용 학습 설정 관리
- 학습 모니터링 및 시각화
- 모델 저장/로드 유틸리티
- Google Colab 환경 지원
"""

import os
import sys
import torch

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
    
    # 네트워크 설정 - CNN만 지원
    'network_type': 'layered',  # 'layered' only
    'observation_type': 'layered',  # CNN 전용
    'use_dueling': True,
    'use_double_dqn': True,
    'use_prioritized_replay': True,
    
    # 학습 모니터링
    'print_interval': 100,
    'save_interval': 500,
    'eval_interval': 200,
    'eval_episodes': 10
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
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # observation_type 강제 설정
    config['observation_type'] = 'layered'
    config['network_type'] = 'layered'
    
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
    
    print(f"✅ 디렉토리 설정 완료: {base_path}")
    return directories

def is_colab_environment():
    """
    Google Colab 환경인지 확인
    
    Returns:
        bool: Colab 환경 여부
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_drive():
    """
    Google Drive 마운트 (Colab 환경에서만)
    
    Returns:
        bool: 마운트 성공 여부
    """
    if not is_colab_environment():
        print("⚠️ Colab 환경이 아닙니다.")
        return False
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive 마운트 완료")
        return True
    except Exception as e:
        print(f"❌ Google Drive 마운트 실패: {e}")
        return False

def get_device_info():
    """
    현재 디바이스 정보 출력
    """
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 디바이스: {torch.cuda.get_device_name()}")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        print("CPU 모드로 실행됩니다.")

# 패키지 정보
__all__ = [
    'DEFAULT_TRAINING_CONFIG',
    'get_training_config',
    'setup_directories',
    'is_colab_environment',
    'mount_drive',
    'get_device_info'
]

print("✅ Train 패키지 로드 완료 - CNN-DQN 전용 학습 환경")