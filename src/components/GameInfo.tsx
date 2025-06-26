import React from 'react';
import { GameState } from '@/types/game';

interface GameInfoProps {
  gameState: GameState;
  isPlaying: boolean;
  speed: number;
  className?: string;
}

export const GameInfo: React.FC<GameInfoProps> = ({
  gameState,
  isPlaying,
  speed,
  className = ''
}) => {
  return (
    <div className={`bg-white p-6 rounded-lg shadow-lg ${className}`}>
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800">{gameState.score.toLocaleString()}</div>
          <div className="text-sm text-gray-600">점수</div>
        </div>
        
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">{gameState.highest}</div>
          <div className="text-sm text-gray-600">최고 타일</div>
        </div>
        
        <div className="text-center">
          <div className="text-lg font-semibold text-blue-600">{gameState.steps}</div>
          <div className="text-sm text-gray-600">스텝</div>
        </div>
        
        <div className="text-center">
          <div className="text-lg font-semibold text-green-600">
            {gameState.board.flat().filter(cell => cell === 0).length}
          </div>
          <div className="text-sm text-gray-600">빈 칸</div>
        </div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex justify-between items-center text-sm">
          <span className="text-gray-600">상태:</span>
          <span className={`font-semibold ${
            gameState.gameOver ? 'text-red-600' : 
            isPlaying ? 'text-green-600' : 'text-gray-600'
          }`}>
            {gameState.gameOver ? '게임 종료' : 
             isPlaying ? `플레이 중 (${speed}x)` : '대기 중'}
          </span>
        </div>
      </div>
    </div>
  );
};