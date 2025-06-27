import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowUp, faArrowRight, faArrowDown, faArrowLeft, faRobot } from '@fortawesome/free-solid-svg-icons';
import { GameAction } from '@/types/game';

interface QValuesDisplayProps {
  qValues: number[] | null;
  selectedAction: GameAction | null;
  className?: string;
}

export const QValuesDisplay: React.FC<QValuesDisplayProps> = ({
  qValues,
  selectedAction,
  className = ''
}) => {
  const actionNames = ['Up', 'Right', 'Down', 'Left'];
  const actionIcons = [faArrowUp, faArrowRight, faArrowDown, faArrowLeft];

  if (!qValues) {
    return (
      <div 
        className={`p-5 rounded-2xl shadow-2xl backdrop-blur-sm flex flex-col ${className}`}
        style={{ 
          background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
          boxShadow: '0 20px 40px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.8)'
        }}
      >
        <h3 
          className="text-lg font-bold mb-4 uppercase tracking-wide text-center"
          style={{ color: '#776e65' }}
        >
          AI predictions
        </h3>
        <div 
          className="flex-1 flex items-center justify-center text-center p-4 rounded-xl"
          style={{ 
            color: '#b59d87',
            background: 'linear-gradient(145deg, #f8f5f0, #f0ede6)',
            border: '2px dashed rgba(181,157,135,0.3)'
          }}
        >
          <div>
            <div className="text-3xl mb-3">
              <FontAwesomeIcon icon={faRobot} style={{ color: '#b59d87' }} />
            </div>
            <div className="text-sm font-medium">게임이 시작되면 AI의 예측값이 표시됩니다</div>
          </div>
        </div>
      </div>
    );
  }

  const maxQValue = Math.max(...qValues);
  const minQValue = Math.min(...qValues);
  const range = maxQValue - minQValue;

  return (
    <div 
      className={`p-5 rounded-2xl shadow-2xl backdrop-blur-sm flex flex-col ${className}`}
      style={{ 
        background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
        boxShadow: '0 20px 40px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.8)'
      }}
    >
      <h3 
        className="text-lg font-bold mb-4 uppercase tracking-wide text-center"
        style={{ color: '#776e65' }}
      >
        AI Predictions
      </h3>
      
      <div className="flex-1 flex flex-col space-y-3 min-h-0">
        {qValues.map((qValue, index) => {
          const normalizedValue = range > 0 ? (qValue - minQValue) / range : 0;
          const isSelected = selectedAction === index;
          
          return (
            <div 
              key={index} 
              className={`p-3 rounded-xl transition-all duration-300 ${isSelected ? 'transform scale-105' : ''}`}
              style={{
                background: isSelected ? 
                  'linear-gradient(145deg, #f0ede6, #e8e3db)' : 
                  'linear-gradient(145deg, #faf8f3, #f5f2eb)',
                border: isSelected ? '2px solid #8f7a66' : '1px solid rgba(206,189,166,0.3)',
                boxShadow: isSelected ? 
                  '0 6px 20px rgba(143,122,102,0.2)' : 
                  '0 2px 8px rgba(0,0,0,0.05)'
              }}
            >
              <div className="flex items-center space-x-3">
                <div 
                  className="w-8 h-8 rounded-xl flex items-center justify-center text-sm font-bold transition-all duration-300"
                  style={{
                    background: isSelected ? 
                      'linear-gradient(145deg, #9f8a76, #8f7a66)' : 
                      'linear-gradient(145deg, #e5d5c3, #d4c2ac)',
                    boxShadow: isSelected ? 
                      '0 3px 12px rgba(143,122,102,0.3), inset 0 1px 0 rgba(255,255,255,0.2)' : 
                      '0 2px 6px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
                  }}
                >
                  <FontAwesomeIcon 
                    icon={actionIcons[index]} 
                    style={{ 
                      color: isSelected ? 'white' : '#776e65',
                      fontSize: '14px'
                    }} 
                  />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex justify-between items-center mb-2">
                    <span 
                      className="text-sm font-bold truncate"
                      style={{ color: isSelected ? '#8f7a66' : '#776e65' }}
                    >
                      {actionNames[index]}
                    </span>
                    <span 
                      className="text-xs font-mono font-bold px-2 py-1 rounded-lg ml-2"
                      style={{ 
                        color: '#776e65',
                        background: 'rgba(206,189,166,0.2)'
                      }}
                    >
                      {qValue.toFixed(3)}
                    </span>
                  </div>
                  
                  <div 
                    className="w-full rounded-full h-2 overflow-hidden"
                    style={{ background: 'rgba(206,189,166,0.3)' }}
                  >
                    <div
                      className="h-2 rounded-full transition-all duration-500 ease-out"
                      style={{
                        width: `${normalizedValue * 100}%`,
                        background: isSelected ? 
                          'linear-gradient(90deg, #9f8a76, #8f7a66)' : 
                          'linear-gradient(90deg, #c4b59f, #b59d87)',
                        boxShadow: isSelected ? '0 0 6px rgba(143,122,102,0.4)' : 'none'
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {selectedAction !== null && (
        <div 
          className="mt-4 p-3 rounded-xl"
          style={{ 
            background: 'linear-gradient(145deg, #f8f5f0, #f0ede6)',
            border: '2px solid rgba(143,122,102,0.2)'
          }}
        >
          <div className="text-center">
            <div className="text-xs font-bold uppercase tracking-wide mb-1" style={{ color: '#b59d87' }}>
              선택된 액션
            </div>
            <div className="text-lg font-bold flex items-center justify-center space-x-2" style={{ color: '#8f7a66' }}>
              <span>{actionNames[selectedAction]}</span>
              <FontAwesomeIcon 
                icon={actionIcons[selectedAction]} 
                style={{ 
                  color: '#8f7a66',
                  fontSize: '16px'
                }} 
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};