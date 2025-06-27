import React from 'react';
import { GameState } from '@/types/game';

interface GameInfoProps {
  gameState: GameState;
  errorCount?: number;
  className?: string;
}

export const GameInfo: React.FC<GameInfoProps> = ({
  gameState,
  errorCount = 0,
  className = ''
}) => {
  return (
    <div className={`${className}`}>
      {/* 모든 게임 정보 - 보드 크기에 맞춘 컴팩트한 디자인 */}
      <div className="flex gap-2 justify-center">
        {/* SCORE */}
        <div 
          className="px-3 py-2 rounded-xl text-center backdrop-blur-sm"
          style={{ 
            width: '100px',
            background: 'linear-gradient(145deg, #c4a688, #a8956f)',
            boxShadow: '0 4px 16px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2)'
          }}
        >
          <div className="text-xs font-bold uppercase tracking-wider" style={{ color: '#f2efea' }}>
            SCORE
          </div>
          <div className="text-lg font-black mt-1" style={{ color: '#ffffff', textShadow: '0 1px 2px rgba(0,0,0,0.2)' }}>
            {gameState.score.toLocaleString()}
          </div>
        </div>
        
        {/* BEST */}
        <div 
          className="px-3 py-2 rounded-xl text-center backdrop-blur-sm"
          style={{ 
            width: '100px',
            background: 'linear-gradient(145deg, #c4a688, #a8956f)',
            boxShadow: '0 4px 16px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2)'
          }}
        >
          <div className="text-xs font-bold uppercase tracking-wider" style={{ color: '#f2efea' }}>
            BEST
          </div>
          <div className="text-lg font-black mt-1" style={{ color: '#ffffff', textShadow: '0 1px 2px rgba(0,0,0,0.2)' }}>
            {gameState.highest}
          </div>
        </div>

        {/* STEPS */}
        <div 
          className="px-3 py-2 rounded-xl text-center backdrop-blur-sm"
          style={{ 
            width: '100px',
            background: 'linear-gradient(145deg, #e5d5c3, #d4c2ac)',
            boxShadow: '0 4px 16px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
          }}
        >
          <div className="text-xs font-bold uppercase tracking-wider" style={{ color: '#8d7c6a' }}>
            STEPS
          </div>
          <div className="text-lg font-black mt-1" style={{ color: '#6d5d4a', textShadow: '0 1px 1px rgba(255,255,255,0.3)' }}>
            {gameState.steps.toLocaleString()}
          </div>
        </div>
        
        {/* EMPTY */}
        <div 
          className="px-3 py-2 rounded-xl text-center backdrop-blur-sm"
          style={{ 
            width: '100px',
            background: 'linear-gradient(145deg, #e5d5c3, #d4c2ac)',
            boxShadow: '0 4px 16px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
          }}
        >
          <div className="text-xs font-bold uppercase tracking-wider" style={{ color: '#8d7c6a' }}>
            EMPTY
          </div>
          <div className="text-lg font-black mt-1" style={{ color: '#6d5d4a', textShadow: '0 1px 1px rgba(255,255,255,0.3)' }}>
            {gameState.board.flat().filter(cell => cell === 0).length}
          </div>
        </div>
      </div>

      {/* 에러 상태 표시 (필요시) */}
      {errorCount > 0 && (
        <div 
          className="text-center p-2 rounded-xl backdrop-blur-sm mt-3"
          style={{ 
            background: 'linear-gradient(145deg, #f67c5f, #f65e3b)',
            boxShadow: '0 8px 24px rgba(246,94,59,0.3)'
          }}
        >
          <div className="text-sm font-bold text-white">
            ⚠️ 오류 발생 횟수: {errorCount}/5
          </div>
        </div>
      )}
    </div>
  );
};