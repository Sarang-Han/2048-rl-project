import React from 'react';
import { GameState } from '@/types/game';

interface GameBoardProps {
  gameState: GameState;
  className?: string;
}

export const GameBoard: React.FC<GameBoardProps> = ({ gameState, className = '' }) => {
  const getTileStyle = (value: number) => {
    const styles: { [key: number]: { bg: string; text: string; fontSize: string; shadow?: string; textShadow?: string } } = {
      0: { bg: 'transparent', text: 'text-transparent', fontSize: 'text-2xl' },
      2: { bg: '#eee4da', text: '#776e65', fontSize: 'text-2xl', shadow: '0 2px 8px rgba(0,0,0,0.1)', textShadow: '0 1px 1px rgba(255,255,255,0.3)' },
      4: { bg: '#ede0c8', text: '#776e65', fontSize: 'text-2xl', shadow: '0 2px 8px rgba(0,0,0,0.1)', textShadow: '0 1px 1px rgba(255,255,255,0.3)' },
      8: { bg: '#f2b179', text: '#ffffff', fontSize: 'text-2xl', shadow: '0 2px 12px rgba(242,177,121,0.4)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      16: { bg: '#f59563', text: '#ffffff', fontSize: 'text-2xl', shadow: '0 2px 12px rgba(245,149,99,0.4)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      32: { bg: '#f67c5f', text: '#ffffff', fontSize: 'text-2xl', shadow: '0 2px 12px rgba(246,124,95,0.4)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      64: { bg: '#f65e3b', text: '#ffffff', fontSize: 'text-2xl', shadow: '0 2px 16px rgba(246,94,59,0.5)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      128: { bg: '#edcf72', text: '#ffffff', fontSize: 'text-xl', shadow: '0 4px 20px rgba(237,207,114,0.6)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      256: { bg: '#edcc61', text: '#ffffff', fontSize: 'text-xl', shadow: '0 4px 20px rgba(237,204,97,0.6)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      512: { bg: '#edc850', text: '#ffffff', fontSize: 'text-xl', shadow: '0 4px 24px rgba(237,200,80,0.7)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      1024: { bg: '#edc53f', text: '#ffffff', fontSize: 'text-lg', shadow: '0 6px 28px rgba(237,197,63,0.8)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
      2048: { bg: '#edc22e', text: '#ffffff', fontSize: 'text-lg', shadow: '0 8px 32px rgba(237,194,46,0.9)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' },
    };
    
    return styles[value] || { bg: '#edc22e', text: '#ffffff', fontSize: 'text-base', shadow: '0 8px 32px rgba(237,194,46,0.9)', textShadow: '0 1px 2px rgba(0,0,0,0.2)' };
  };

  // 크기 상수 정의
  const BOARD_SIZE = 440;
  const PADDING = 24;
  const GAP = 12;
  const TILE_SIZE = (BOARD_SIZE - PADDING * 2 - GAP * 3) / 4; // 정확한 타일 크기 계산: 89px

  return (
    <div className={`relative ${className}`}>
      {/* 게임보드 컨테이너 - 완전 고정 크기 */}
      <div 
        className="relative rounded-2xl shadow-2xl overflow-hidden"
        style={{ 
          width: `${BOARD_SIZE}px`,
          height: `${BOARD_SIZE}px`,
          background: 'linear-gradient(145deg, #c4b59f, #a89a82)',
          boxShadow: '0 20px 40px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2)',
          padding: `${PADDING}px`
        }}
      >
        {/* 배경 그리드 */}
        <div 
          className="absolute grid grid-cols-4"
          style={{
            top: `${PADDING}px`,
            left: `${PADDING}px`,
            width: `${BOARD_SIZE - PADDING * 2}px`,
            height: `${BOARD_SIZE - PADDING * 2}px`,
            gap: `${GAP}px`
          }}
        >
          {Array.from({ length: 16 }, (_, i) => (
            <div
              key={i}
              className="rounded-xl"
              style={{ 
                width: `${TILE_SIZE}px`,
                height: `${TILE_SIZE}px`,
                background: 'rgba(206, 189, 166, 0.4)' 
              }}
            />
          ))}
        </div>
        
        {/* 타일들 - 절대 위치 */}
        <div 
          className="absolute"
          style={{
            top: `${PADDING}px`,
            left: `${PADDING}px`,
            width: `${BOARD_SIZE - PADDING * 2}px`,
            height: `${BOARD_SIZE - PADDING * 2}px`
          }}
        >
          {gameState.board.map((row, i) =>
            row.map((cell, j) => {
              const tileStyle = getTileStyle(cell);
              
              // 각 타일의 정확한 위치 계산
              const tileX = j * (TILE_SIZE + GAP);
              const tileY = i * (TILE_SIZE + GAP);
              
              return (
                <div
                  key={`${i}-${j}`}
                  className={`
                    absolute rounded-xl flex items-center justify-center
                    font-black transition-all duration-300 ease-out
                    ${tileStyle.fontSize}
                  `}
                  style={{ 
                    left: `${tileX}px`,
                    top: `${tileY}px`,
                    width: `${TILE_SIZE}px`,
                    height: `${TILE_SIZE}px`,
                    background: tileStyle.bg,
                    color: tileStyle.text,
                    boxShadow: tileStyle.shadow || 'none',
                    textShadow: tileStyle.textShadow || 'none',
                    transform: cell !== 0 ? 'scale(1)' : 'scale(0.8)',
                    opacity: cell !== 0 ? 1 : 0,
                    zIndex: cell !== 0 ? 10 : 1
                  }}
                >
                  {cell !== 0 ? cell.toLocaleString() : ''}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};