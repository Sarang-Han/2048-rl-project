import React from 'react';
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
  const actionNames = ['위', '오른쪽', '아래', '왼쪽'];
  const actionIcons = ['⬆️', '➡️', '⬇️', '⬅️'];

  if (!qValues) {
    return (
      <div 
        className={`p-8 rounded-2xl shadow-2xl backdrop-blur-sm ${className}`}
        style={{ 
          background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
          boxShadow: '0 20px 40px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.8)'
        }}
      >
        <h3 
          className="text-xl font-bold mb-6 uppercase tracking-wide text-center"
          style={{ color: '#776e65' }}
        >
          AI 예측값
        </h3>
        <div 
          className="text-center py-8 px-4 rounded-xl"
          style={{ 
            color: '#b59d87',
            background: 'linear-gradient(145deg, #f8f5f0, #f0ede6)',
            border: '2px dashed rgba(181,157,135,0.3)'
          }}
        >
          <div className="text-4xl mb-3">🤖</div>
          <div className="font-medium">게임이 시작되면 AI의 예측값이 표시됩니다</div>
        </div>
      </div>
    );
  }

  const maxQValue = Math.max(...qValues);
  const minQValue = Math.min(...qValues);
  const range = maxQValue - minQValue;

  return (
    <div 
      className={`p-8 rounded-2xl shadow-2xl backdrop-blur-sm ${className}`}
      style={{ 
        background: 'linear-gradient(145deg, #ffffff, #f8f5f0)',
        boxShadow: '0 20px 40px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.8)'
      }}
    >
      <h3 
        className="text-xl font-bold mb-6 uppercase tracking-wide text-center"
        style={{ color: '#776e65' }}
      >
        AI 예측값
      </h3>
      
      <div className="space-y-4">
        {qValues.map((qValue, index) => {
          const normalizedValue = range > 0 ? (qValue - minQValue) / range : 0;
          const isSelected = selectedAction === index;
          
          return (
            <div 
              key={index} 
              className={`p-4 rounded-xl transition-all duration-300 ${isSelected ? 'transform scale-105' : ''}`}
              style={{
                background: isSelected ? 
                  'linear-gradient(145deg, #f0ede6, #e8e3db)' : 
                  'linear-gradient(145deg, #faf8f3, #f5f2eb)',
                border: isSelected ? '2px solid #8f7a66' : '1px solid rgba(206,189,166,0.3)',
                boxShadow: isSelected ? 
                  '0 8px 24px rgba(143,122,102,0.2)' : 
                  '0 2px 8px rgba(0,0,0,0.05)'
              }}
            >
              <div className="flex items-center space-x-4">
                <div 
                  className="w-12 h-12 rounded-2xl flex items-center justify-center text-xl font-bold transition-all duration-300"
                  style={{
                    background: isSelected ? 
                      'linear-gradient(145deg, #9f8a76, #8f7a66)' : 
                      'linear-gradient(145deg, #e5d5c3, #d4c2ac)',
                    color: isSelected ? 'white' : '#776e65',
                    boxShadow: isSelected ? 
                      '0 4px 16px rgba(143,122,102,0.3), inset 0 1px 0 rgba(255,255,255,0.2)' : 
                      '0 2px 8px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.3)'
                  }}
                >
                  {actionIcons[index]}
                </div>
                
                <div className="flex-1">
                  <div className="flex justify-between items-center mb-3">
                    <span 
                      className="text-lg font-bold"
                      style={{ color: isSelected ? '#8f7a66' : '#776e65' }}
                    >
                      {actionNames[index]}
                    </span>
                    <span 
                      className="text-sm font-mono font-bold px-3 py-1 rounded-lg"
                      style={{ 
                        color: '#776e65',
                        background: 'rgba(206,189,166,0.2)'
                      }}
                    >
                      {qValue.toFixed(3)}
                    </span>
                  </div>
                  
                  <div 
                    className="w-full rounded-full h-3 overflow-hidden"
                    style={{ background: 'rgba(206,189,166,0.3)' }}
                  >
                    <div
                      className="h-3 rounded-full transition-all duration-500 ease-out"
                      style={{
                        width: `${normalizedValue * 100}%`,
                        background: isSelected ? 
                          'linear-gradient(90deg, #9f8a76, #8f7a66)' : 
                          'linear-gradient(90deg, #c4b59f, #b59d87)',
                        boxShadow: isSelected ? '0 0 8px rgba(143,122,102,0.4)' : 'none'
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
          className="mt-6 p-4 rounded-xl"
          style={{ 
            background: 'linear-gradient(145deg, #f8f5f0, #f0ede6)',
            border: '2px solid rgba(143,122,102,0.2)'
          }}
        >
          <div className="text-center">
            <div className="text-sm font-bold uppercase tracking-wide mb-2" style={{ color: '#b59d87' }}>
              선택된 액션
            </div>
            <div className="text-2xl font-bold" style={{ color: '#8f7a66' }}>
              {actionNames[selectedAction]} {actionIcons[selectedAction]}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};