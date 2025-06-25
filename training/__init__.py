"""
2048 DQN 학습 패키지

이 패키지는 2048 게임 환경과 DQN 에이전트 학습을 위한 모든 구성요소를 포함합니다.
- environment: 2048 게임 환경
- models: DQN 에이전트, 신경망, 재생 버퍼
- train: 학습 스크립트 및 설정
"""

import sys
import os

# 현재 패키지 경로를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 패키지 정보
__all__ = [
    'environment',
    'models',
    'train'
]

# 버전 정보
__version__ = "1.0.0"

def get_package_info():
    """패키지 정보 반환"""
    return {
        "training_package_version": __version__,
        "python_path": current_dir,
        "available_modules": __all__
    }