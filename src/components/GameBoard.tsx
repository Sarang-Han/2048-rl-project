import React from 'react';
import { GameState } from '@/types/game';
import { theme } from '@/lib/theme';

interface GameBoardProps {
  gameState: GameState;
  className?: string;
}

const GameBoardComponent: React.FC<GameBoardProps> = ({ gameState, className = '' }) => {
  const getTileStyle = (value: number) => {
    const styles: { [key: number]: { bg: string; text: string; fontSize: string; boxShadow?: string } } = {
      0: { bg: 'transparent', text: 'text-transparent', fontSize: 'text-xl' },
      // 🔥 기존 색상 유지하면서 그림자만 추가
      2: { 
        bg: '#eee4da', 
        text: '#776e65', 
        fontSize: 'text-3xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
      },
      4: { 
        bg: '#ede0c8', 
        text: '#776e65', 
        fontSize: 'text-3xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
      },
      8: { 
        bg: '#f2b179', 
        text: '#ffffff', 
        fontSize: 'text-3xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      16: { 
        bg: '#f59563', 
        text: '#ffffff', 
        fontSize: 'text-3xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      32: { 
        bg: '#f67c5f', 
        text: '#ffffff', 
        fontSize: 'text-3xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      64: { 
        bg: '#f65e3b', 
        text: '#ffffff', 
        fontSize: 'text-3xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      128: { 
        bg: '#edcf72', 
        text: '#ffffff', 
        fontSize: 'text-2xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
      },
      256: { 
        bg: '#edcc61', 
        text: '#ffffff', 
        fontSize: 'text-2xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
      },
      512: { 
        bg: '#edc850', 
        text: '#ffffff', 
        fontSize: 'text-2xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
      },
      1024: { 
        bg: '#edc53f', 
        text: '#ffffff', 
        fontSize: 'text-xl',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
      },
      2048: { 
        bg: '#edc22e', 
        text: '#ffffff', 
        fontSize: 'text-xl',
        boxShadow: `0 3px 6px rgba(119, 110, 101, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.5), inset 0 0 0 2px rgba(255, 215, 0, 0.3)`
      },
      // 🔥 2048보다 높은 값들에 대한 새로운 스타일 (그림자 추가)
      4096: { 
        bg: '#ff6b6b', 
        text: '#ffffff', 
        fontSize: 'text-lg',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      8192: { 
        bg: '#4ecdc4', 
        text: '#ffffff', 
        fontSize: 'text-lg',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      16384: { 
        bg: '#45b7d1', 
        text: '#ffffff', 
        fontSize: 'text-base',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      32768: { 
        bg: '#96ceb4', 
        text: '#ffffff', 
        fontSize: 'text-base',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      65536: { 
        bg: '#feca57', 
        text: '#ffffff', 
        fontSize: 'text-sm',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
      131072: { 
        bg: '#ff9ff3', 
        text: '#ffffff', 
        fontSize: 'text-sm',
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.3)`
      },
    };
    
    // 정의된 스타일이 있으면 사용
    if (styles[value]) {
      return styles[value];
    }
    
    // 🔥 매우 높은 값들에 대한 동적 스타일 생성 (그림자 포함)
    if (value > 131072) {
      const fontSize = value > 1000000 ? 'text-xs' : 'text-sm';
      return { 
        bg: '#2c2c54', 
        text: '#ffffff', 
        fontSize,
        boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.2)`
      };
    }
    
    // 기본 fallback (그림자 포함)
    return { 
      bg: '#edc22e', 
      text: '#ffffff', 
      fontSize: 'text-base',
      boxShadow: `${theme.shadows.tile.main}, inset 0 1px 0 rgba(255, 255, 255, 0.4)`
    };
  };

  return (
    <div className={`relative select-none ${className}`}>
      {/* 🔥 게임보드 컨테이너 */}
      <div 
        className="flex items-center justify-center rounded-xl"
        style={{ 
          width: `${theme.spacing.boardSize}px`,
          height: `${theme.spacing.boardSize}px`,
          backgroundColor: theme.colors.board.frame,
          padding: `${theme.spacing.gap}px`,
          boxShadow: `
            0 1px 0 rgba(255, 255, 255, 0.3),
            0 -1px 0 rgba(0, 0, 0, 0.2),
            1px 0 0 rgba(255, 255, 255, 0.2),
            -1px 0 0 rgba(0, 0, 0, 0.15),
            0 4px 8px rgba(119, 110, 101, 0.2)
          `
        }}
      >
        {/* 그리드 컨테이너: 배경 셀과 타일을 모두 담는 역할 */}
        <div
          className="relative grid grid-cols-4 grid-rows-4 w-full h-full"
          style={{
            gap: `${theme.spacing.gap}px`,
          }}
        >
          {/* 🔥 배경 셀들 - 더 선명한 3D 효과 */}
          {Array.from({ length: 16 }).map((_, index) => (
            <div
              key={index}
              className="rounded-lg"
              style={{ 
                backgroundColor: theme.colors.board.cellEmpty,
                // 🔥 더 선명한 그림자와 하이라이트
                boxShadow: `
                  inset 0 1px 0 #b0a491,
                  inset 0 -1px 0 #d8cfc1,
                  inset 1px 0 0 rgba(0, 0, 0, 0.15),
                  inset -1px 0 0 rgba(255, 255, 255, 0.15)
                `
              }}
            />
          ))}
          
          {/* 🔥 타일들: 그리드 위에 절대 위치로 배치, 3D 효과 추가 */}
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
                    boxShadow: cell !== 0 ? tileStyle.boxShadow : 'none',
                    transform: cell !== 0 ? 'scale(1)' : 'scale(0)',
                    opacity: cell !== 0 ? 1 : 0,
                    zIndex: cell !== 0 ? 10 : 1,
                    transition: 'all 0.2s ease-in-out, box-shadow 0.1s ease-in-out'
                  }}
                  onMouseEnter={(e) => {
                    if (cell !== 0) {
                      e.currentTarget.style.boxShadow = tileStyle.boxShadow?.replace(
                        theme.shadows.tile.main, 
                        theme.shadows.tile.hover
                      ) || theme.shadows.tile.hover;
                      e.currentTarget.style.transform = 'scale(1.02) translateY(-1px)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (cell !== 0) {
                      e.currentTarget.style.boxShadow = tileStyle.boxShadow || theme.shadows.tile.main;
                      e.currentTarget.style.transform = 'scale(1)';
                    }
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