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
    <div className={`bg-white p-6 rounded-lg shadow-lg ${className}`}>
      <div className="space-y-4">
        {/* 재생/일시정지/리셋 버튼 */}
        <div className="flex gap-3">
          <button
            onClick={isPlaying ? onPause : onPlay}
            disabled={!isModelReady}
            className={`
              flex-1 py-3 px-4 rounded-md font-semibold text-white transition-colors
              ${isPlaying 
                ? 'bg-orange-500 hover:bg-orange-600' 
                : 'bg-green-500 hover:bg-green-600'
              }
              ${!isModelReady ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            {isPlaying ? '⏸️ 일시정지' : '▶️ 시작'}
          </button>
          
          <button
            onClick={onReset}
            className="px-4 py-3 bg-gray-500 hover:bg-gray-600 text-white font-semibold rounded-md transition-colors"
          >
            🔄 리셋
          </button>
        </div>

        {/* 속도 조절 */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            게임 속도
          </label>
          <div className="grid grid-cols-4 gap-2">
            {speedOptions.map((speedOption) => (
              <button
                key={speedOption}
                onClick={() => onSpeedChange(speedOption)}
                className={`
                  py-2 px-3 rounded-md text-sm font-medium transition-colors
                  ${speed === speedOption
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }
                `}
              >
                {speedOption}x
              </button>
            ))}
          </div>
        </div>

        {/* 모델 상태 */}
        <div className="pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">AI 모델:</span>
            <span className={`font-semibold ${
              isModelReady ? 'text-green-600' : 'text-orange-600'
            }`}>
              {isModelReady ? '✅ 준비됨' : '⏳ 로딩 중...'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};