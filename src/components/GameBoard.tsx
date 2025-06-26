import React from 'react';
import { GameState } from '@/types/game';

interface GameBoardProps {
  gameState: GameState;
  className?: string;
}

export const GameBoard: React.FC<GameBoardProps> = ({ gameState, className = '' }) => {
  const getTileColor = (value: number): string => {
    const colors: { [key: number]: string } = {
      0: 'bg-gray-200 text-gray-400',
      2: 'bg-gray-100 text-gray-800',
      4: 'bg-gray-200 text-gray-800',
      8: 'bg-orange-200 text-orange-800',
      16: 'bg-orange-300 text-orange-900',
      32: 'bg-orange-400 text-white',
      64: 'bg-red-400 text-white',
      128: 'bg-yellow-300 text-white font-bold',
      256: 'bg-yellow-400 text-white font-bold',
      512: 'bg-yellow-500 text-white font-bold',
      1024: 'bg-purple-500 text-white font-bold',
      2048: 'bg-red-500 text-white font-bold',
    };
    
    return colors[value] || 'bg-purple-600 text-white font-bold';
  };

  const getTileSize = (value: number): string => {
    if (value >= 1024) return 'text-xs';
    if (value >= 100) return 'text-sm';
    return 'text-lg';
  };

  return (
    <div className={`bg-gray-300 p-4 rounded-lg shadow-lg ${className}`}>
      <div className="grid grid-cols-4 gap-2">
        {gameState.board.map((row, i) =>
          row.map((cell, j) => (
            <div
              key={`${i}-${j}`}
              className={`
                w-16 h-16 rounded-md flex items-center justify-center
                font-bold transition-all duration-200 ease-in-out
                ${getTileColor(cell)} ${getTileSize(cell)}
                ${cell !== 0 ? 'transform scale-100' : ''}
              `}
            >
              {cell !== 0 ? cell : ''}
            </div>
          ))
        )}
      </div>
    </div>
  );
};