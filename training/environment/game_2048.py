import gym
from gym import spaces
import numpy as np
import random
from typing import Tuple, Dict, Any, Optional, List
import itertools

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

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
    2048 게임 Gym 환경
    
    Action Space: Discrete(4)
        0: 위로 이동
        1: 오른쪽으로 이동  
        2: 아래로 이동
        3: 왼쪽으로 이동
    
    Observation Space: Box(0, 1, (4, 4, 16), dtype=np.float32) - 원-핫 인코딩
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, size: int = 4, max_steps: int = 10000):
        super(Game2048Env, self).__init__()
        
        # 입력 검증
        if size <= 0:
            raise ValueError("Size must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        self.size = size
        self.squares = self.size * self.size
        self.max_steps = max_steps
        
        # 액션 공간
        self.action_space = spaces.Discrete(4)
        
        # 관찰 공간 설정
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.size, self.size, self.squares), 
            dtype=np.float32
        )
        
        # 게임 상태 초기화
        self.board: Optional[np.ndarray] = None
        self.score: int = 0
        self.steps: int = 0
        self._game_initialized: bool = False
        
        # 보상 설정
        self.illegal_move_reward = 0.0
        self.max_tile: Optional[int] = None
        self.reward_range = (0.0, float(2**self.squares))
    
    def set_illegal_move_reward(self, reward: float) -> None:
        """불법 이동에 대한 보상/페널티 설정"""
        self.illegal_move_reward = float(reward)
        self.reward_range = (
            min(self.illegal_move_reward, 0.0), 
            float(2**self.squares)
        )
    
    def set_max_tile(self, max_tile: Optional[int]) -> None:
        """게임을 종료할 최대 타일 설정"""
        if max_tile is not None:
            if not isinstance(max_tile, int) or max_tile <= 0:
                raise ValueError("max_tile must be a positive integer or None")
            if max_tile & (max_tile - 1) != 0:
                raise ValueError("max_tile must be a power of 2 (e.g., 64, 128, 256, 512, 1024, 2048)")
        self.max_tile = max_tile
    
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
        """한 스텝 실행 (액션 마스킹 적용)"""
        if not self._game_initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        self.steps += 1
        info = {'illegal_move': False}
        done = False
        
        try:
            # 움직임 시도
            move_score = self._move(action)
            self.score += move_score
            
            # 새 타일 추가
            self._add_random_tile()
            
            # 게임 종료 확인
            done = self._is_game_over()
            reward = self._calculate_reward(move_score)
            
        except IllegalMove:
            # 액션 마스킹이 적용되면 이 부분은 실행되지 않아야 함
            info['illegal_move'] = True
            reward = self.illegal_move_reward
            print(f"⚠️ Warning: Illegal move {action} attempted! Action masking should prevent this.")
        
        # 게임 종료 조건 확인
        if not done:
            if self.max_tile is not None and np.max(self.board) >= self.max_tile:
                done = True
            elif self.steps >= self.max_steps:
                done = True
        
        info.update({
            'score': int(self.score),
            'highest': int(np.max(self.board)),
            'empty_cells': len(self._get_empty_cells()),
            'steps': self.steps,
            'valid_actions': self.get_valid_actions()
        })
        
        return self._get_observation(), reward, done, info

    def render(self, mode: str = 'human') -> Optional[str]:
        """게임 상태 시각화"""
        if not self._game_initialized:
            print("Game not initialized. Call reset() first.")
            return None
            
        if mode == 'ansi':
            return self._render_ansi()
        elif mode == 'human':
            print(self._render_ansi())
            return None
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
        """향상된 보상 함수"""
        # 가중치
        W_MERGE = 1.0
        W_EMPTY = 2.7
        W_MONO = 1.0
        W_SMOOTH = 0.1

        # 1. 합병 점수
        merge_reward = np.log2(move_score) if move_score > 0 else 0

        # 2. 빈 타일 보상
        empty_cells = len(self._get_empty_cells())
        empty_reward = np.log(empty_cells) if empty_cells > 0 else 0

        # 3. 단조성 보상
        mono_reward = self._calculate_monotonicity()

        # 4. 평탄성 보상
        smooth_reward = self._calculate_smoothness()

        total_reward = (
            W_MERGE * merge_reward +
            W_EMPTY * empty_reward +
            W_MONO * mono_reward +
            W_SMOOTH * smooth_reward
        )
        
        return float(total_reward)

    def _calculate_monotonicity(self) -> float:
        """보드의 단조성 계산"""
        monotonicity_score = 0
        
        # 행 단조성
        for i in range(self.size):
            row = self.board[i, :]
            row_values = row[row != 0]
            if len(row_values) > 1:
                log_vals = np.log2(row_values)
                monotonicity_score += max(np.sum(np.diff(log_vals)), np.sum(-np.diff(log_vals)))

        # 열 단조성
        for j in range(self.size):
            col = self.board[:, j]
            col_values = col[col != 0]
            if len(col_values) > 1:
                log_vals = np.log2(col_values)
                monotonicity_score += max(np.sum(np.diff(log_vals)), np.sum(-np.diff(log_vals)))
                
        return monotonicity_score

    def _calculate_smoothness(self) -> float:
        """보드의 평탄성 계산"""
        smoothness_score = 0
        log_board = np.log2(self.board, where=self.board > 0, out=np.zeros_like(self.board, dtype=float))

        # 수평 평탄성
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] != 0 and self.board[i, j + 1] != 0:
                    smoothness_score -= abs(log_board[i, j] - log_board[i, j + 1])

        # 수직 평탄성
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] != 0 and self.board[i + 1, j] != 0:
                    smoothness_score -= abs(log_board[i, j] - log_board[i + 1, j])
                    
        return smoothness_score
    
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
        
        if direction:
            combined_row.reverse()
        
        return (combined_row, move_score)
    
    def _combine(self, shifted_row: List[int]) -> Tuple[List[int], int]:
        """같은 타일을 병합"""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        
        for p in pairwise(shifted_row):
            if skip:
                combined_row[output_index] = p[1]
                output_index += 1
                skip = False
            elif p[0] == p[1]:
                combined_row[output_index] = 2 * p[0]
                move_score += combined_row[output_index]
                output_index += 1
                skip = True
            else:
                combined_row[output_index] = p[0]
                output_index += 1
        
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]
        
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
        """현재 보드 상태 반환 (테스트용)"""
        if not self._game_initialized:
            raise RuntimeError("Game not initialized. Call reset() first.")
        return self.board.copy()

    def set_board(self, board: np.ndarray) -> None:
        """보드 상태 설정 (테스트용)"""
        if board.shape != (self.size, self.size):
            raise ValueError(f"Board shape must be ({self.size}, {self.size})")
        self.board = board.copy()
        self._game_initialized = True

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
        
        try:
            self._move(action)
            return True
        except IllegalMove:
            return False
        finally:
            self.board = original_board

    def get_action_mask(self) -> np.ndarray:
        """액션 마스크를 numpy 배열로 반환"""
        if not self._game_initialized:
            return np.zeros(4, dtype=bool)
        
        mask = np.zeros(4, dtype=bool)
        valid_actions = self.get_valid_actions()
        
        for action in valid_actions:
            mask[action] = True
        
        return mask

# 테스트 함수
def test_action_masking():
    """액션 마스킹 테스트"""
    print("🧪 액션 마스킹 테스트")
    
    env = Game2048Env()
    state = env.reset()
    
    # 몇 번의 스텝 실행
    for step in range(10):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("게임 종료 - 유효한 액션 없음")
            break
            
        action = valid_actions[0]  # 첫 번째 유효한 액션 선택
        next_state, reward, done, info = env.step(action)
        
        print(f"Step {step+1}: Action {action}, Reward {reward:.2f}, Valid actions: {valid_actions}")
        
        if done:
            print("게임 종료")
            break
    
    print("✅ 액션 마스킹 테스트 완료")

if __name__ == "__main__":
    test_action_masking()