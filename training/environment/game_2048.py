import gym
from gym import spaces
import numpy as np
import random
from typing import Tuple, Dict, Any, Optional, List

class IllegalMove(Exception):
    """불가능한 움직임을 시도할 때 발생하는 예외"""
    pass

def stack(flat: np.ndarray, layers: int = 16) -> np.ndarray:
    """
    Convert an [4, 4] representation into [4, 4, layers] with one layer for each value.
    
    Args:
        flat: 4x4 보드 배열
        layers: 레이어 수 (기본값: 16)
    
    Returns:
        4x4x16 원-핫 인코딩된 배열
    """
    representation = 2 ** (np.arange(layers, dtype=int) + 1)
    layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)
    layered = np.where(layered == representation, 1, 0)
    return layered.astype(np.float32)

class Game2048Env(gym.Env):
    """
    2048 게임 Gym 환경 (액션 마스킹 지원)
    
    Action Space: Discrete(4) - 0: 위, 1: 오른쪽, 2: 아래, 3: 왼쪽
    Observation Space: Box(0, 1, (4, 4, 16), dtype=np.float32) - 원-핫 인코딩
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, size: int = 4, max_steps: int = 10000):
        super(Game2048Env, self).__init__()
        
        if size <= 0:
            raise ValueError("Size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        self.size = size
        self.squares = self.size * self.size
        self.max_steps = max_steps
        
        # 액션/관찰 공간
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.size, self.size, self.squares), 
            dtype=np.float32
        )
        
        # 게임 상태
        self.board: Optional[np.ndarray] = None
        self.score: int = 0
        self.steps: int = 0
        self._game_initialized: bool = False
        
        # 보상 설정
        # self.reward_range = (0.0, float(2**self.squares))
    
    def reset(self) -> np.ndarray:
        """환경 초기화"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.steps = 0
        self._game_initialized = True
        
        # 초기 타일 2개 추가
        self._add_random_tile()
        self._add_random_tile()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """한 스텝 실행"""
        if not self._game_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        self.steps += 1
        info = {'illegal_move': False}
        done = False
        
        try:
            move_score = self._move(action)
            self.score += move_score
            self._add_random_tile()
            done = self._is_game_over()
            reward = self._calculate_reward(move_score)
            
        except IllegalMove:
            info['illegal_move'] = True
            reward = 0.0
        
        # 게임 종료 조건 (스텝 수 제한)
        if not done and self.steps >= self.max_steps:
            done = True
        
        info.update({
            'score': int(self.score),
            'highest': int(np.max(self.board)),
            'empty_cells': len(self._get_empty_cells()),
            'steps': self.steps,
            'valid_actions': self.get_valid_actions(),
            'move_score': move_score if 'move_score' in locals() else 0
        })
        
        return self._get_observation(), reward, done, info

    def render(self, mode: str = 'human') -> Optional[str]:
        """게임 상태 시각화"""
        if not self._game_initialized:
            print("Game not initialized. Call reset() first.")
            return None
            
        output = self._render_ansi()
        if mode == 'human':
            print(output)
            return None
        elif mode == 'ansi':
            return output
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")

    def _render_ansi(self) -> str:
        """ANSI 형태로 보드 렌더링"""
        if not self._game_initialized:
            return "Game not initialized. Call reset() first.\n"
        
        s = f"Score: {self.score} | Steps: {self.steps}\n"
        s += f"Highest: {int(np.max(self.board))} | Empty: {len(self._get_empty_cells())}\n"
        s += "-" * 25 + "\n"
        
        for row in self.board:
            s += "|"
            for cell in row:
                if cell == 0:
                    s += "    ."
                else:
                    s += f"{cell:5d}"
            s += " |\n"
        
        s += "-" * 25 + "\n"
        return s

    def _calculate_reward(self, move_score: float) -> float:
        """휴리스틱 보상 함수 (기본형)"""
        # 가중치
        W_MERGE = 1.0
        W_EMPTY = 2.7

        # 1. 합병 점수
        merge_reward = np.log2(max(move_score, 1)) if move_score > 0 else 0

        # 2. 빈 타일 보상
        empty_cells = len(self._get_empty_cells())
        empty_reward = np.log(max(empty_cells, 1)) if empty_cells > 0 else 0

        total_reward = (
            W_MERGE * merge_reward +
            W_EMPTY * empty_reward
        )
        
        return float(total_reward)
    
    def _add_random_tile(self) -> bool:
        """빈 자리에 랜덤 타일 추가"""
        empty_cells = self._get_empty_cells()
        if not empty_cells:
            return False
        
        i, j = random.choice(empty_cells)
        self.board[i][j] = 2 if random.random() < 0.9 else 4
        return True

    def _get_empty_cells(self) -> List[Tuple[int, int]]:
        """빈 셀들의 좌표 반환"""
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
    
    def _move(self, direction: int) -> float:
        """지정된 방향으로 보드 이동"""
        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two
        
        if dir_mod_two == 0:
            # 수직 이동
            for j in range(self.size):
                col = [self.board[i][j] for i in range(self.size)]
                (new_col, score) = self._shift(col, shift_direction)
                move_score += score
                for i in range(self.size):
                    if self.board[i][j] != new_col[i]:
                        changed = True
                    self.board[i][j] = new_col[i]
        else:
            # 수평 이동
            for i in range(self.size):
                row = [self.board[i][j] for j in range(self.size)]
                (new_row, score) = self._shift(row, shift_direction)
                move_score += score
                for j in range(self.size):
                    if self.board[i][j] != new_row[j]:
                        changed = True
                    self.board[i][j] = new_row[j]
        
        if not changed:
            raise IllegalMove(f"Move {direction} is not valid")
        
        return move_score
    
    def _shift(self, row: List[int], direction: int) -> Tuple[List[int], int]:
        """한 행을 이동하며 병합"""
        shifted_row = [i for i in row if i != 0]
        
        if direction:
            shifted_row.reverse()
        
        (combined_row, move_score) = self._combine(shifted_row)
        
        while len(combined_row) < self.size:
            combined_row.append(0)
        
        if direction:
            combined_row.reverse()
        
        return (combined_row, move_score)
    
    def _combine(self, shifted_row: List[int]) -> Tuple[List[int], int]:
        """같은 타일을 병합"""
        if not shifted_row:
            return ([0] * self.size, 0)
        
        move_score = 0
        combined_row = []
        i = 0
        
        while i < len(shifted_row):
            if (i + 1 < len(shifted_row) and 
                shifted_row[i] == shifted_row[i + 1] and 
                shifted_row[i] != 0):
                # 같은 타일 병합
                merged_value = 2 * shifted_row[i]
                combined_row.append(merged_value)
                move_score += merged_value
                i += 2
            else:
                combined_row.append(shifted_row[i])
                i += 1
        
        return (combined_row, move_score)
    
    def _is_game_over(self) -> bool:
        """게임 종료 여부 확인"""
        if self._get_empty_cells():
            return False
        
        # 인접한 같은 숫자 확인
        for i in range(self.size):
            for j in range(self.size):
                current = self.board[i][j]
                if j < self.size - 1 and current == self.board[i][j + 1]:
                    return False
                if i < self.size - 1 and current == self.board[i + 1][j]:
                    return False
        
        return True

    def _get_observation(self) -> np.ndarray:
        """현재 상태를 관찰 공간에 맞게 변환"""
        if not self._game_initialized:
            raise RuntimeError("Game not initialized. Call reset() first.")
        
        return stack(self.board.astype(np.float32), layers=self.squares)

    def get_board(self) -> np.ndarray:
        """현재 보드 상태 반환"""
        if not self._game_initialized:
            raise RuntimeError("Game not initialized. Call reset() first.")
        return self.board.copy()

    def get_valid_actions(self) -> List[int]:
        """현재 상태에서 유효한 액션들을 반환"""
        if not self._game_initialized:
            return []
        
        valid_actions = []
        for action in range(4):
            if self._is_valid_move(action):
                valid_actions.append(action)
        
        return valid_actions

    def _is_valid_move(self, action: int) -> bool:
        """특정 액션이 유효한지 확인"""
        original_board = self.board.copy()
        original_score = self.score
        
        try:
            self._move(action)
            return True
        except IllegalMove:
            return False
        finally:
            self.board = original_board
            self.score = original_score

    def get_action_mask(self) -> np.ndarray:
        """액션 마스크를 numpy 배열로 반환"""
        if not self._game_initialized:
            return np.zeros(4, dtype=bool)
        
        mask = np.zeros(4, dtype=bool)
        valid_actions = self.get_valid_actions()
        
        for action in valid_actions:
            mask[action] = True
        
        return mask