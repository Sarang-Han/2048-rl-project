import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlay, faPause, faRotateRight, faCircle } from '@fortawesome/free-solid-svg-icons';
import { GameSpeed } from '@/types/game';
import { theme } from '@/lib/theme';

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

const GameControls: React.FC<GameControlsProps> = ({
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
      className={`p-5 rounded-xl flex flex-col ${className}`}
      style={{ 
        backgroundColor: theme.colors.controls.background,
        border: `${theme.borders.medium} ${theme.colors.controls.border}`
      }}
    >
      <div className="flex flex-col h-full">
        <div>
          <h3 
            className="text-lg font-bold mb-4 tracking-wide text-center"
            style={{ color: theme.colors.primary.text }}
          >
            Game Control
          </h3>
        </div>

        {/* 메인 컨트롤 버튼들 */}
        <div className="space-y-3 mb-4">
          <button
            onClick={isPlaying ? onPause : onPlay}
            disabled={!isModelReady}
            className="w-full py-3 px-4 rounded-lg font-bold text-sm transition-all duration-300 transform flex items-center justify-center space-x-2"
            style={{
              backgroundColor: !isModelReady ? '#d0c4b0' : 
                            isPlaying ? theme.colors.controls.button.danger : 
                            theme.colors.controls.button.primary,
              color: 'white',
              border: `${theme.borders.thin} ${!isModelReady ? '#b0a090' : 
                      isPlaying ? theme.colors.controls.button.dangerHover : 
                      theme.colors.controls.button.primaryHover}`,
              cursor: !isModelReady ? 'not-allowed' : 'pointer',
              opacity: !isModelReady ? 0.6 : 1
            }}
            onMouseOver={(e) => {
              if (isModelReady) {
                e.currentTarget.style.backgroundColor = isPlaying ? 
                  theme.colors.controls.button.dangerHover : 
                  theme.colors.controls.button.primaryHover;
              }
            }}
            onMouseOut={(e) => {
              if (isModelReady) {
                e.currentTarget.style.backgroundColor = isPlaying ? 
                  theme.colors.controls.button.danger : 
                  theme.colors.controls.button.primary;
              }
            }}
          >
            <FontAwesomeIcon 
              icon={isPlaying ? faPause : faPlay} 
              style={{ fontSize: '14px' }}
            />
            <span>{isPlaying ? 'Pause' : 'AI Play'}</span>
          </button>
          
          <button
            onClick={onReset}
            className="w-full py-3 px-4 font-bold text-sm rounded-lg transition-all duration-300 flex items-center justify-center space-x-2"
            style={{
              backgroundColor: theme.colors.controls.button.secondary,
              color: theme.colors.primary.text,
              border: `${theme.borders.thin} ${theme.colors.controls.button.secondaryHover}`
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = theme.colors.controls.button.secondaryHover;
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = theme.colors.controls.button.secondary;
            }}
          >
            <FontAwesomeIcon 
              icon={faRotateRight} 
              style={{ fontSize: '14px' }}
            />
            <span>New Game</span>
          </button>
        </div>

        {/* 속도 조절 */}
        <div className="mb-4">
          <label 
            className="block text-sm font-bold mb-3 tracking-wide text-center"
            style={{ color: theme.colors.primary.text }}
          >
            Speed
          </label>
          <div className="grid grid-cols-4 gap-2">
            {speedOptions.map((speedOption) => (
              <button
                key={speedOption}
                onClick={() => onSpeedChange(speedOption)}
                className="py-2 px-2 rounded-md text-xs font-bold transition-all duration-300"
                style={{
                  backgroundColor: speed === speedOption ? 
                    theme.colors.controls.button.primary : 
                    theme.colors.controls.button.secondary,
                  color: speed === speedOption ? 'white' : theme.colors.primary.text,
                  border: `${theme.borders.thin} ${speed === speedOption ? 
                    theme.colors.controls.button.primaryHover : 
                    theme.colors.controls.button.secondaryHover}`
                }}
              >
                {speedOption}x
              </button>
            ))}
          </div>
        </div>

        {/* 모델 상태 */}
        <div 
          className="pt-4 border-t-2 mt-auto"
          style={{ borderColor: theme.colors.controls.border }}
        >
          <div className="flex items-center justify-between text-sm">
            <span style={{ color: theme.colors.primary.text }} className="font-bold">AI Model</span>
            <div className="flex items-center space-x-2">
              <FontAwesomeIcon 
                icon={faCircle} 
                style={{ 
                  color: isModelReady ? theme.colors.status.success : theme.colors.status.warning,
                  fontSize: '8px'
                }}
              />
              <span 
                className="font-bold text-xs"
                style={{ color: isModelReady ? theme.colors.status.success : theme.colors.status.warning }}
              >
                {isModelReady ? 'READY' : 'LOADING...'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

GameControls.displayName = 'GameControls';

export { GameControls };