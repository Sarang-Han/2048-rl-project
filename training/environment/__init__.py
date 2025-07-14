"""
2048 게임 환경 패키지

OpenAI Gym 스타일의 2048 게임 환경을 제공합니다.
- CNN을 위한 layered 관찰 타입 지원: (4, 4, 16)
- 표준 Gym 인터페이스: reset(), step(), render()
- 보상 함수 및 게임 종료 조건 포함
"""

from .game_2048 import Game2048Env, IllegalMove, stack, pairwise

# 패키지에서 공개할 클래스/함수들
__all__ = [
    'Game2048Env',
    'IllegalMove',
    'stack',
    'pairwise'
]

# 환경 설정 - CNN 전용
DEFAULT_ENV_CONFIG = {
    'observation_type': 'layered',  # 'layered' only
    'max_steps': 1000,
    'render_mode': 'human'
}

def create_env(**kwargs):
    """
    편의 함수: 2048 환경 생성 (CNN 전용)
    
    Args:
        **kwargs: 추가 환경 설정
    
    Returns:
        Game2048Env: 초기화된 게임 환경 (layered 관찰 타입)
    """
    config = DEFAULT_ENV_CONFIG.copy()
    config.update(kwargs)
    # observation_type을 layered로 강제 설정
    config['observation_type'] = 'layered'
    return Game2048Env(**config)

# 지원되는 관찰 타입 - CNN만 지원
SUPPORTED_OBSERVATION_TYPES = ['layered']

# 액션 정보
ACTION_MEANINGS = {
    0: 'UP',
    1: 'RIGHT', 
    2: 'DOWN',
    3: 'LEFT'
}

print(f"✅ Environment 패키지 로드 완료 - 지원 관찰 타입: {SUPPORTED_OBSERVATION_TYPES}")