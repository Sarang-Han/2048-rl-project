import React from 'react';
import { GameSpeed } from '@/types/game';

interface GameControlsProps {
  isPlaying: boolean;
  isModelReady: boolean;
  speed: GameSpeed;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSpeedChange: (speed: GameSpeed) => void;
  className?: string;
}

export const GameControls: React.FC<GameControlsProps> = ({
  isPlaying,
  isModelReady,
  speed,
  onPlay,
  onPause,
  onReset,
  onSpeedChange,
  className = ''
}) => {
  const speedOptions: GameSpeed[] = [0.5, 1, 2, 4];

  return (
    <div 
      className={`p-8 rounded-2xl shadow-2xl backdrop-blur-sm ${className}`}
      style={{ 
        background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
        boxShadow: '0 20px 40px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.8)'
      }}
    >
      <div className="space-y-8">
        <div>
          <h3 
            className="text-xl font-bold mb-6 uppercase tracking-wide text-center"
            style={{ color: '#776e65' }}
          >
            게임 컨트롤
          </h3>
        </div>

        {/* 메인 컨트롤 버튼들 */}
        <div className="space-y-4">
          <button
            onClick={isPlaying ? onPause : onPlay}
            disabled={!isModelReady}
            className="w-full py-4 px-6 rounded-2xl font-bold text-lg transition-all duration-300 transform"
            style={{
              background: !isModelReady ? '#d0c4b0' : 
                         isPlaying ? 'linear-gradient(145deg, #f67c5f, #f65e3b)' : 
                         'linear-gradient(145deg, #9f8a76, #8f7a66)',
              color: 'white',
              boxShadow: !isModelReady ? 'none' : '0 8px 24px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2)',
              cursor: !isModelReady ? 'not-allowed' : 'pointer',
              opacity: !isModelReady ? 0.6 : 1
            }}
            onMouseEnter={(e) => {
              if (isModelReady) {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 12px 32px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.2)';
              }
            }}
            onMouseLeave={(e) => {
              if (isModelReady) {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2)';
              }
            }}
          >
            {isPlaying ? '⏸️ 일시정지' : '▶️ AI 시작'}
          </button>
          
          <button
            onClick={onReset}
            className="w-full py-4 px-6 font-bold text-lg rounded-2xl transition-all duration-300 transform"
            style={{
              background: 'linear-gradient(145deg, #d4c2ac, #c4b59f)',
              color: '#776e65',
              boxShadow: '0 6px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 8px 28px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.3)';
              e.currentTarget.style.background = 'linear-gradient(145deg, #e5d5c3, #d4c2ac)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)';
              e.currentTarget.style.background = 'linear-gradient(145deg, #d4c2ac, #c4b59f)';
            }}
          >
            🔄 새 게임
          </button>
        </div>

        {/* 속도 조절 */}
        <div>
          <label 
            className="block text-base font-bold mb-4 uppercase tracking-wide text-center"
            style={{ color: '#776e65' }}
          >
            게임 속도
          </label>
          <div className="grid grid-cols-4 gap-3">
            {speedOptions.map((speedOption) => (
              <button
                key={speedOption}
                onClick={() => onSpeedChange(speedOption)}
                className="py-3 px-3 rounded-xl text-sm font-bold transition-all duration-300 transform"
                style={{
                  background: speed === speedOption ? 
                    'linear-gradient(145deg, #9f8a76, #8f7a66)' : 
                    'linear-gradient(145deg, #e5d5c3, #d4c2ac)',
                  color: speed === speedOption ? 'white' : '#776e65',
                  boxShadow: speed === speedOption ? 
                    '0 4px 16px rgba(143,122,102,0.3), inset 0 1px 0 rgba(255,255,255,0.2)' : 
                    '0 2px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
                }}
                onMouseEnter={(e) => {
                  if (speed !== speedOption) {
                    e.currentTarget.style.background = 'linear-gradient(145deg, #c4b59f, #b59d87)';
                    e.currentTarget.style.color = 'white';
                    e.currentTarget.style.transform = 'translateY(-1px)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (speed !== speedOption) {
                    e.currentTarget.style.background = 'linear-gradient(145deg, #e5d5c3, #d4c2ac)';
                    e.currentTarget.style.color = '#776e65';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }
                }}
              >
                {speedOption}x
              </button>
            ))}
          </div>
        </div>

        {/* 모델 상태 */}
        <div 
          className="pt-6 border-t-2"
          style={{ borderColor: 'rgba(206, 189, 166, 0.3)' }}
        >
          <div className="flex items-center justify-between text-base">
            <span style={{ color: '#776e65' }} className="font-bold">AI 모델</span>
            <div className="flex items-center space-x-2">
              <div 
                className="w-3 h-3 rounded-full"
                style={{ 
                  background: isModelReady ? 
                    'linear-gradient(145deg, #4ade80, #22c55e)' : 
                    'linear-gradient(145deg, #fbbf24, #f59e0b)',
                  boxShadow: isModelReady ? 
                    '0 0 8px rgba(34,197,94,0.4)' : 
                    '0 0 8px rgba(245,158,11,0.4)'
                }}
              />
              <span 
                className="font-bold text-sm"
                style={{ color: isModelReady ? '#22c55e' : '#f59e0b' }}
              >
                {isModelReady ? '준비됨' : '로딩 중...'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};