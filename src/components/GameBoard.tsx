import React from 'react';
import { GameState } from '@/types/game';
import { theme } from '@/lib/theme';

interface GameBoardProps {
  gameState: GameState;
  className?: string;
}

const GameBoardComponent: React.FC<GameBoardProps> = ({ gameState, className = '' }) => {
  const getTileStyle = (value: number) => {
    const baseStyles: { [key: number]: { bg: string; text: string; fontSize: string } } = {
      0: { bg: 'transparent', text: 'text-transparent', fontSize: 'text-xl' },
      2: { bg: '#eee4da', text: '#776e65', fontSize: 'text-3xl' },
      4: { bg: '#ede0c8', text: '#776e65', fontSize: 'text-3xl' },
      8: { bg: '#f2b179', text: '#ffffff', fontSize: 'text-3xl' },
      16: { bg: '#f59563', text: '#ffffff', fontSize: 'text-3xl' },
      32: { bg: '#f67c5f', text: '#ffffff', fontSize: 'text-3xl' },
      64: { bg: '#f65e3b', text: '#ffffff', fontSize: 'text-3xl' },
      128: { bg: '#edcf72', text: '#ffffff', fontSize: 'text-2xl' },
      256: { bg: '#edcc61', text: '#ffffff', fontSize: 'text-2xl' },
      512: { bg: '#edc850', text: '#ffffff', fontSize: 'text-2xl' },
      1024: { bg: '#edc53f', text: '#ffffff', fontSize: 'text-xl' },
      2048: { bg: '#edc22e', text: '#ffffff', fontSize: 'text-xl' }
    };
    
    if (baseStyles[value]) return baseStyles[value];
    
    // 높은 값들
    if (value > 2048) {
      return { 
        bg: value > 8192 ? '#2c2c54' : '#ff6b6b', 
        text: '#ffffff', 
        fontSize: value > 16384 ? 'text-sm' : 'text-lg'
      };
    }
    
    return { bg: '#edc22e', text: '#ffffff', fontSize: 'text-base' };
  };

  return (
    <div className={`relative select-none ${className}`}>
      <div 
        className="flex items-center justify-center rounded-xl"
        style={{ 
          width: `${theme.spacing.boardSize}px`,
          height: `${theme.spacing.boardSize}px`,
          backgroundColor: theme.colors.board.frame,
          padding: `${theme.spacing.gap}px`,
          boxShadow: '0 4px 8px rgba(119, 110, 101, 0.2)'
        }}
      >
        <div className="relative grid grid-cols-4 grid-rows-4 w-full h-full" style={{ gap: `${theme.spacing.gap}px` }}>
          {/* 배경 셀들 */}
          {Array.from({ length: 16 }).map((_, index) => (
            <div key={index} className="rounded-lg" style={{ backgroundColor: theme.colors.board.cellEmpty }} />
          ))}
          
          {/* 타일들 */}
          <div className="absolute top-0 left-0 w-full h-full grid grid-cols-4 grid-rows-4" style={{ gap: `${theme.spacing.gap}px` }}>
            {gameState.board.flat().map((cell, index) => {
              const tileStyle = getTileStyle(cell);
              
              return (
                <div
                  key={index}
                  className={`flex items-center justify-center rounded-lg font-black transition-all duration-200 ${tileStyle.fontSize}`}
                  style={{ 
                    backgroundColor: tileStyle.bg,
                    color: tileStyle.text,
                    boxShadow: cell !== 0 ? theme.shadows.tile.main : 'none',
                    transform: cell !== 0 ? 'scale(1)' : 'scale(0)',
                    opacity: cell !== 0 ? 1 : 0,
                    zIndex: cell !== 0 ? 10 : 1
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

export const GameBoard = React.memo(GameBoardComponent, (prevProps, nextProps) => {
  const prev = prevProps.gameState;
  const next = nextProps.gameState;
  
  if (prev.score !== next.score || prev.steps !== next.steps || 
      prev.highest !== next.highest || prev.gameOver !== next.gameOver) {
    return false;
  }

  return prev.board.flat().every((cell, i) => cell === next.board.flat()[i]);
});

GameBoard.displayName = 'GameBoard';