"""
2048 CNN-DQN 학습 패키지
"""

import os
import torch

# 학습 기본 설정
DEFAULT_TRAINING_CONFIG = {
    # 기본 설정
    'episodes': 2000,
    'max_steps_per_episode': 1000,
    'device': 'auto',
    
    # DQN 하이퍼파라미터
    'buffer_size': 100000,
    'batch_size': 64,
    'lr': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 50000,
    'target_update': 1000,
    
    # 학습 모니터링
    'log_interval': 10,
    'save_interval': 500,
    'eval_interval': 100,
    'eval_episodes': 10,
    'plot_interval': 50
}

def get_training_config(custom_config=None):
    """학습 설정 반환"""
    config = DEFAULT_TRAINING_CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    
    # device 자동 설정
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return config

def setup_directories(base_path='/content/drive/MyDrive/2048_models'):
    """학습에 필요한 디렉토리 생성"""
    directories = [
        base_path,
        os.path.join(base_path, 'checkpoints'),
        os.path.join(base_path, 'best_models'),
        os.path.join(base_path, 'onnx_models')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"✅ 디렉토리 설정 완료: {base_path}")
    return directories

def is_colab_environment():
    """Google Colab 환경인지 확인"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_drive():
    """Google Drive 마운트 (Colab 환경에서만)"""
    if not is_colab_environment():
        print("Colab 환경이 아닙니다.")
        return False
    
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive 마운트 완료")
        return True
    except Exception as e:
        print(f"Google Drive 마운트 실패: {e}")
        return False

def get_device_info():
    """현재 디바이스 정보 출력"""
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 디바이스: {torch.cuda.get_device_name()}")
    else:
        print("CPU 모드로 실행됩니다.")

__all__ = [
    'DEFAULT_TRAINING_CONFIG',
    'get_training_config', 
    'setup_directories',
    'is_colab_environment',
    'mount_drive',
    'get_device_info'
]

print("Train 패키지 로드 완료")