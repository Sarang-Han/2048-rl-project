"""
2048 강화학습 프로젝트

이 프로젝트는 2048 게임을 플레이하는 DQN 에이전트를 구현합니다.
- CNN 기반 아키텍처 (layered observation)
- DNN 기반 아키텍처 (flat observation)
- Double DQN, Dueling DQN, 우선순위 재생 버퍼 지원
"""

__version__ = "1.0.0"
__author__ = "2048 DQN Team"
__description__ = "2048 Game Deep Q-Learning Agent"

# 패키지 정보
PACKAGE_INFO = {
    "name": "2048-rl-project",
    "version": __version__,
    "description": __description__,
    "author": __author__
}