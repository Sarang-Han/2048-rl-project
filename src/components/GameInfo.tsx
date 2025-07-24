import React from 'react';
import { GameState } from '@/types/game';
import { theme } from '@/lib/theme';

interface GameInfoProps {
  gameState: GameState;
  errorCount?: number;
  className?: string;
}

const GameInfoComponent: React.FC<GameInfoProps> = ({
  gameState,
  errorCount = 0,
  className = ''
}) => {
  const emptyCount = gameState.board.flat().filter(cell => cell === 0).length;
  
  return (
    <div className={`${className}`}>
      {/* 🔥 메인 게임 정보 - 4개 카드만 표시 */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        <InfoCard
          label="SCORE"
          value={gameState.score.toLocaleString()}
          type="primary"
        />
        <InfoCard
          label="BEST"
          value={gameState.highest.toString()}
          type="primary"
        />
        <InfoCard
          label="STEPS"
          value={gameState.steps.toLocaleString()}
          type="secondary"
        />
        <InfoCard
          label="EMPTY"
          value={emptyCount.toString()}
          type="secondary"
        />
      </div>

      {/* 🔥 에러 상태 (조건부 표시) */}
      {errorCount > 0 && (
        <div className="bg-red-100 border border-red-300 rounded-xl p-3 text-center">
          <div className="text-sm font-bold text-red-700">
            ⚠️ 오류 발생: {errorCount}/3
          </div>
          <div className="text-xs text-red-600 mt-1">
            연속 오류 시 게임이 중단됩니다
          </div>
        </div>
      )}
    </div>
  );
};

// 🔥 재사용 가능한 InfoCard 컴포넌트
interface InfoCardProps {
  label: string;
  value: string;
  type: 'primary' | 'secondary' | 'info';
  icon?: string;
}

const InfoCard: React.FC<InfoCardProps> = ({ label, value, type, icon }) => {
  const getCardStyle = () => {
    switch (type) {
      case 'primary':
        return {
          background: theme.colors.controls.button.primary,
          textColor: '#ffffff',
          labelColor: '#f2efea'
        };
      case 'secondary':
        return {
          background: theme.colors.controls.button.secondary,
          textColor: '#6d5d4a',
          labelColor: '#8d7c6a'
        };
      case 'info':
        return {
          background: 'linear-gradient(145deg, #e3f2fd, #bbdefb)',
          textColor: '#1565c0',
          labelColor: '#1976d2'
        };
    }
  };

  const style = getCardStyle();

  return (
    <div 
      className="px-3 py-2 rounded-xl text-center backdrop-blur-sm"
      style={{ 
        background: style.background,
        boxShadow: theme.colors.board.shadow
      }}
    >
      <div className="flex items-center justify-center space-x-1">
        {icon && <span className="text-xs">{icon}</span>}
        <div 
          className="text-xs font-bold uppercase tracking-wider"
          style={{ color: style.labelColor }}
        >
          {label}
        </div>
      </div>
      <div 
        className="text-lg font-black mt-1"
        style={{ 
          color: style.textColor, 
          textShadow: type === 'primary' ? '0 1px 2px rgba(0,0,0,0.2)' : '0 1px 1px rgba(255,255,255,0.3)'
        }}
      >
        {value}
      </div>
    </div>
  );
};

export const GameInfo = React.memo(GameInfoComponent, (prevProps, nextProps) => 
  prevProps.gameState.score === nextProps.gameState.score &&
  prevProps.gameState.highest === nextProps.gameState.highest &&
  prevProps.gameState.steps === nextProps.gameState.steps &&
  prevProps.errorCount === nextProps.errorCount
);