import React from 'react';
import { GameState } from '@/types/game';
import { theme } from '@/lib/theme';

interface GameBoardProps {
  gameState: GameState;
  className?: string;
}

const GameBoardComponent: React.FC<GameBoardProps> = ({ gameState, className = '' }) => {
  const getTileStyle = (value: number) => {
    const styles: { [key: number]: { bg: string; text: string; fontSize: string; border?: string } } = {
      0: { bg: 'transparent', text: 'text-transparent', fontSize: 'text-xl' },
      2: { bg: '#eee4da', text: '#776e65', fontSize: 'text-3xl', border: `${theme.borders.thin} #d4c2ac` },
      4: { bg: '#ede0c8', text: '#776e65', fontSize: 'text-3xl', border: `${theme.borders.thin} #c4b59f` },
      8: { bg: '#f2b179', text: '#ffffff', fontSize: 'text-3xl', border: `${theme.borders.thin} #e09f65` },
      16: { bg: '#f59563', text: '#ffffff', fontSize: 'text-3xl', border: `${theme.borders.thin} #e3834f` },
      32: { bg: '#f67c5f', text: '#ffffff', fontSize: 'text-3xl', border: `${theme.borders.thin} #e46a4b` },
      64: { bg: '#f65e3b', text: '#ffffff', fontSize: 'text-3xl', border: `${theme.borders.thin} #e24c27` },
      128: { bg: '#edcf72', text: '#ffffff', fontSize: 'text-2xl', border: `${theme.borders.thin} #d9bb5e` },
      256: { bg: '#edcc61', text: '#ffffff', fontSize: 'text-2xl', border: `${theme.borders.thin} #d9b84d` },
      512: { bg: '#edc850', text: '#ffffff', fontSize: 'text-2xl', border: `${theme.borders.thin} #d9b43c` },
      1024: { bg: '#edc53f', text: '#ffffff', fontSize: 'text-xl', border: `${theme.borders.thin} #d9b12b` },
      2048: { bg: '#edc22e', text: '#ffffff', fontSize: 'text-xl', border: `${theme.borders.medium} #d9ae1a` },
      // 🔥 2048보다 높은 값들에 대한 새로운 스타일 추가
      4096: { bg: '#ff6b6b', text: '#ffffff', fontSize: 'text-lg', border: `${theme.borders.medium} #e55555` },
      8192: { bg: '#4ecdc4', text: '#ffffff', fontSize: 'text-lg', border: `${theme.borders.medium} #3ba99c` },
      16384: { bg: '#45b7d1', text: '#ffffff', fontSize: 'text-base', border: `${theme.borders.medium} #3498c4` },
      32768: { bg: '#96ceb4', text: '#ffffff', fontSize: 'text-base', border: `${theme.borders.medium} #7fb89a` },
      65536: { bg: '#feca57', text: '#ffffff', fontSize: 'text-sm', border: `${theme.borders.medium} #dea945` },
      131072: { bg: '#ff9ff3', text: '#ffffff', fontSize: 'text-sm', border: `${theme.borders.medium} #e588d6` },
    };
    
    // 정의된 스타일이 있으면 사용
    if (styles[value]) {
      return styles[value];
    }
    
    // 🔥 매우 높은 값들에 대한 동적 스타일 생성
    if (value > 131072) {
      // 값이 클수록 더 작은 폰트와 특별한 색상 사용
      const fontSize = value > 1000000 ? 'text-xs' : 'text-sm';
      return { 
        bg: '#2c2c54', 
        text: '#ffffff', 
        fontSize, 
        border: `${theme.borders.medium} #1a1a35` 
      };
    }
    
    // 기본 fallback (혹시 놓친 값들을 위해)
    return { bg: '#edc22e', text: '#ffffff', fontSize: 'text-base', border: `${theme.borders.medium} #d9ae1a` };
  };

  return (
    <div className={`relative select-none ${className}`}>
      {/* 게임보드 컨테이너: Flexbox를 사용하여 내부 그리드를 중앙에 배치 */}
      <div 
        className="flex items-center justify-center rounded-xl"
        style={{ 
          width: `${theme.spacing.boardSize}px`,
          height: `${theme.spacing.boardSize}px`,
          backgroundColor: theme.colors.board.background,
          border: `${theme.borders.thick} ${theme.colors.board.border}`,
          padding: `${theme.spacing.gap}px`,
        }}
      >
        {/* 그리드 컨테이너: 배경 셀과 타일을 모두 담는 역할 */}
        <div
          className="relative grid grid-cols-4 grid-rows-4 w-full h-full"
          style={{
            gap: `${theme.spacing.gap}px`,
          }}
        >
          {/* 배경 셀들 */}
          {Array.from({ length: 16 }).map((_, index) => (
            <div
              key={index}
              className="rounded-lg"
              style={{ 
                backgroundColor: theme.colors.board.cellEmpty,
                border: `${theme.borders.thin} ${theme.colors.board.border}`
              }}
            />
          ))}
          
          {/* 타일들: 그리드 위에 절대 위치로 배치 */}
          <div className="absolute top-0 left-0 w-full h-full grid grid-cols-4 grid-rows-4" style={{ gap: `${theme.spacing.gap}px` }}>
            {gameState.board.flat().map((cell, index) => {
              const tileStyle = getTileStyle(cell);
              
              return (
                <div
                  key={index}
                  className={`
                    flex items-center justify-center rounded-lg
                    font-black transition-all duration-200 ease-in-out
                    ${tileStyle.fontSize}
                  `}
                  style={{ 
                    backgroundColor: tileStyle.bg,
                    color: tileStyle.text,
                    border: tileStyle.border || 'none',
                    transform: cell !== 0 ? 'scale(1)' : 'scale(0)',
                    opacity: cell !== 0 ? 1 : 0,
                    zIndex: cell !== 0 ? 10 : 1,
                  }}
                >
                  {cell !== 0 ? cell.toLocaleString() : ''}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

const areGameStatesEqual = (prevState: GameState, nextState: GameState): boolean => {
  // 점수, 스텝, 최고점, 게임오버 상태 비교
  if (
    prevState.score !== nextState.score ||
    prevState.steps !== nextState.steps ||
    prevState.highest !== nextState.highest ||
    prevState.gameOver !== nextState.gameOver
  ) {
    return false;
  }

  // 보드 배열 깊은 비교
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (prevState.board[i][j] !== nextState.board[i][j]) {
        return false;
      }
    }
  }

  return true;
};

export const GameBoard = React.memo(GameBoardComponent, (prevProps, nextProps) => 
  areGameStatesEqual(prevProps.gameState, nextProps.gameState)
);

GameBoard.displayName = 'GameBoard';