import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlay, faPause, faRotateRight, faCircle } from '@fortawesome/free-solid-svg-icons';
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
      className={`p-5 rounded-2xl backdrop-blur-sm flex flex-col ${className}`}
      style={{ 
        background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
        boxShadow: '0 6px 20px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.8)'
      }}
    >
      <div className="flex flex-col h-full">
        <div>
          <h3 
            className="text-lg font-bold mb-4 tracking-wide text-center"
            style={{ color: '#776e65' }}
          >
            Game Control
          </h3>
        </div>

        {/* 메인 컨트롤 버튼들 */}
        <div className="space-y-3 mb-4">
          <button
            onClick={isPlaying ? onPause : onPlay}
            disabled={!isModelReady}
            className="w-full py-3 px-4 rounded-xl font-bold text-sm transition-all duration-300 transform flex items-center justify-center space-x-2"
            style={{
              background: !isModelReady ? '#d0c4b0' : 
                         isPlaying ? 'linear-gradient(145deg, #f67c5f, #f65e3b)' : 
                         'linear-gradient(145deg, #9f8a76, #8f7a66)',
              color: 'white',
              boxShadow: !isModelReady ? 'none' : '0 4px 12px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2)',
              cursor: !isModelReady ? 'not-allowed' : 'pointer',
              opacity: !isModelReady ? 0.6 : 1
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
            className="w-full py-3 px-4 font-bold text-sm rounded-xl transition-all duration-300 flex items-center justify-center space-x-2"
            style={{
              background: 'linear-gradient(145deg, #d4c2ac, #c4b59f)',
              color: '#776e65',
              boxShadow: '0 3px 10px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
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
            style={{ color: '#776e65' }}
          >
            Speed
          </label>
          <div className="grid grid-cols-4 gap-2">
            {speedOptions.map((speedOption) => (
              <button
                key={speedOption}
                onClick={() => onSpeedChange(speedOption)}
                className="py-2 px-2 rounded-lg text-xs font-bold transition-all duration-300"
                style={{
                  background: speed === speedOption ? 
                    'linear-gradient(145deg, #9f8a76, #8f7a66)' : 
                    'linear-gradient(145deg, #e5d5c3, #d4c2ac)',
                  color: speed === speedOption ? 'white' : '#776e65',
                  boxShadow: speed === speedOption ? 
                    '0 2px 8px rgba(143,122,102,0.25), inset 0 1px 0 rgba(255,255,255,0.2)' : 
                    '0 2px 4px rgba(0,0,0,0.08), inset 0 1px 0 rgba(255,255,255,0.3)'
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
          style={{ borderColor: 'rgba(206, 189, 166, 0.3)' }}
        >
          <div className="flex items-center justify-between text-sm">
            <span style={{ color: '#776e65' }} className="font-bold">AI Model</span>
            <div className="flex items-center space-x-2">
              <FontAwesomeIcon 
                icon={faCircle} 
                style={{ 
                  color: isModelReady ? '#22c55e' : '#f59e0b',
                  fontSize: '8px',
                  filter: isModelReady ? 
                    'drop-shadow(0 0 3px rgba(34,197,94,0.4))' : 
                    'drop-shadow(0 0 3px rgba(245,158,11,0.4))'
                }}
              />
              <span 
                className="font-bold text-xs"
                style={{ color: isModelReady ? '#22c55e' : '#f59e0b' }}
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