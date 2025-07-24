export interface GameState {
  board: number[][];
  score: number;
  steps: number;
  highest: number;
  isEmpty: boolean;
  gameOver: boolean;
}

export interface GameInfo {
  score: number;
  highest: number;
  empty_cells: number;
  steps: number;
  illegal_move: boolean;
}

export type GameAction = 0 | 1 | 2 | 3;

export interface ModelPrediction {
  action: GameAction;
  qValues: number[];
}

export type GameSpeed = 0.5 | 1 | 2 | 4;