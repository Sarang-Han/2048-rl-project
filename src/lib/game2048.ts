import { GameState, GameInfo, GameAction } from '@/types/game';

export class Game2048 {
  private board: number[][] = [];
  private score: number = 0;
  private steps: number = 0;
  private gameOver: boolean = false;
  private readonly size: number = 4;

  constructor() {
    this.reset();
  }

  reset(): GameState {
    this.board = Array(this.size).fill(null).map(() => Array(this.size).fill(0));
    this.score = 0;
    this.steps = 0;
    this.gameOver = false;
    
    // 초기 타일 2개 추가
    this.addRandomTile();
    this.addRandomTile();
    
    return this.getState();
  }

  // 🔥 GUI용으로 간소화된 step 메서드 (보상 계산 제거)
  step(action: GameAction): { state: GameState; done: boolean; info: GameInfo } {
    if (this.gameOver) {
      return {
        state: this.getState(),
        done: true,
        info: this.getInfo()
      };
    }

    this.steps++;
    let illegalMove = false;

    try {
      const moveScore = this.move(action);
      this.score += moveScore;
      
      // 새 타일 추가
      if (!this.addRandomTile()) {
        // 타일 추가 실패 - 보드가 가득 참
        this.gameOver = this.isGameOver();
      } else {
        // 게임 종료 확인
        this.gameOver = this.isGameOver();
      }
    } catch {
      illegalMove = true;
      console.warn(`⚠️ Warning: Illegal move ${action} attempted! Action masking should prevent this.`);
    }

    return {
      state: this.getState(),
      done: this.gameOver,
      info: {
        ...this.getInfo(),
        illegal_move: illegalMove
      }
    };
  }

  private move(direction: GameAction): number {
    const { board, score, changed } = this.testMove(direction, this.board);

    if (!changed) {
      throw new Error("No valid moves available");
    }

    this.board = board;
    return score;
  }

  private testMove(direction: GameAction, board: number[][]): { board: number[][], score: number, changed: boolean } {
    let moveScore = 0;
    let changed = false;
    const newBoard = board.map(row => [...row]);

    const dirDivTwo = Math.floor(direction / 2);
    const dirModTwo = direction % 2;
    const shiftDirection = dirModTwo ^ dirDivTwo;

    if (dirModTwo === 0) {
      // Up or down, split into columns
      for (let y = 0; y < this.size; y++) {
        const oldColumn = [];
        for (let x = 0; x < this.size; x++) {
          oldColumn.push(newBoard[x][y]);
        }
        
        const [newColumn, score] = this.shift(oldColumn, shiftDirection);
        moveScore += score;
        
        if (!this.arraysEqual(oldColumn, newColumn)) {
          changed = true;
          for (let x = 0; x < this.size; x++) {
            newBoard[x][y] = newColumn[x];
          }
        }
      }
    } else {
      // Left or right, split into rows
      for (let x = 0; x < this.size; x++) {
        const oldRow = [...newBoard[x]];
        const [newRow, score] = this.shift(oldRow, shiftDirection);
        moveScore += score;
        
        if (!this.arraysEqual(oldRow, newRow)) {
          changed = true;
          newBoard[x] = newRow;
        }
      }
    }

    return { board: newBoard, score: moveScore, changed };
  }

  private shift(row: number[], direction: number): [number[], number] {
    // Shift all non-zero digits
    const shiftedRow = row.filter(cell => cell !== 0);
    
    // Reverse if shifting right
    if (direction) {
      shiftedRow.reverse();
    }
    
    const [combinedRow, moveScore] = this.combine(shiftedRow);
    
    // Reverse back if shifting right
    if (direction) {
      combinedRow.reverse();
    }
    
    return [combinedRow, moveScore];
  }

  private combine(shiftedRow: number[]): [number[], number] {
    let moveScore = 0;
    const combinedRow = Array(this.size).fill(0);
    let skip = false;
    let outputIndex = 0;
    
    for (let i = 0; i < shiftedRow.length - 1; i++) {
      if (skip) {
        skip = false;
        continue;
      }
      
      combinedRow[outputIndex] = shiftedRow[i];
      
      if (shiftedRow[i] === shiftedRow[i + 1]) {
        combinedRow[outputIndex] += shiftedRow[i + 1];
        moveScore += shiftedRow[i] + shiftedRow[i + 1];
        skip = true;
      }
      outputIndex++;
    }
    
    if (shiftedRow.length > 0 && !skip) {
      combinedRow[outputIndex] = shiftedRow[shiftedRow.length - 1];
    }
    
    return [combinedRow, moveScore];
  }

  private addRandomTile(): boolean {
    const emptyCells = this.getEmptyCells();
    if (emptyCells.length === 0) return false;
    
    const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
    const [i, j] = randomCell;
    
    // 90% 확률로 2, 10% 확률로 4
    this.board[i][j] = Math.random() < 0.9 ? 2 : 4;
    return true;
  }

  private getEmptyCells(): [number, number][] {
    const emptyCells: [number, number][] = [];
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        if (this.board[i][j] === 0) {
          emptyCells.push([i, j]);
        }
      }
    }
    return emptyCells;
  }

  private isGameOver(): boolean {
    // 빈 칸이 있으면 게임 계속
    if (this.getEmptyCells().length > 0) return false;
    
    // 인접한 같은 숫자가 있는지 확인
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const current = this.board[i][j];
        // 오른쪽 인접 셀 확인
        if (j < this.size - 1 && this.board[i][j + 1] === current) return false;
        // 아래쪽 인접 셀 확인
        if (i < this.size - 1 && this.board[i + 1][j] === current) return false;
      }
    }
    
    return true;
  }

  private arraysEqual(a: number[], b: number[]): boolean {
    return a.length === b.length && a.every((val, i) => val === b[i]);
  }

  getState(): GameState {
    return {
      board: this.board.map(row => [...row]),
      score: this.score,
      steps: this.steps,
      highest: Math.max(...this.board.flat()),
      isEmpty: this.getEmptyCells().length > 0,
      gameOver: this.gameOver
    };
  }

  private getInfo(): GameInfo {
    return {
      score: this.score,
      highest: Math.max(...this.board.flat()),
      empty_cells: this.getEmptyCells().length,
      steps: this.steps,
      illegal_move: false
    };
  }

  // 🔥 AI 모델용 layered observation 생성
  getObservation(): Float32Array {
    // (4, 4, 16) 형태의 layered observation 생성
    const observation = new Float32Array(4 * 4 * 16);
    
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const value = this.board[i][j];
        if (value > 0) {
          // 2^n 값을 n-1 인덱스로 변환 (2→0, 4→1, 8→2, ...)
          const layerIndex = Math.log2(value) - 1;
          if (layerIndex >= 0 && layerIndex < 16) {
            const index = i * 4 * 16 + j * 16 + layerIndex;
            observation[index] = 1.0;
          }
        }
      }
    }
    
    return observation;
  }

  // 🔥 유효한 액션인지 미리 확인하는 메서드 (성능 최적화)
  isValidAction(action: GameAction): boolean {
    const { changed } = this.testMove(action, this.board);
    return changed;
  }

  // 🔥 가능한 액션들 반환 - 액션 마스킹의 핵심
  getValidActions(): GameAction[] {
    const validActions: GameAction[] = [];
    for (let action = 0; action < 4; action++) {
      if (this.isValidAction(action as GameAction)) {
        validActions.push(action as GameAction);
      }
    }
    return validActions;
  }

  getBoard(): number[][] {
    return this.board.map(row => [...row]);
  }
}