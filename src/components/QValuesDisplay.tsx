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
      <div className={`bg-white p-6 rounded-lg shadow-lg ${className}`}>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">AI 예측값</h3>
        <div className="text-center text-gray-500">
          게임이 시작되면 AI의 예측값이 표시됩니다.
        </div>
      </div>
    );
  }

  const maxQValue = Math.max(...qValues);
  const minQValue = Math.min(...qValues);
  const range = maxQValue - minQValue;

  return (
    <div className={`bg-white p-6 rounded-lg shadow-lg ${className}`}>
      <h3 className="text-lg font-semibold text-gray-800 mb-4">AI 예측값 (Q-Values)</h3>
      
      <div className="space-y-3">
        {qValues.map((qValue, index) => {
          const normalizedValue = range > 0 ? (qValue - minQValue) / range : 0;
          const isSelected = selectedAction === index;
          
          return (
            <div key={index} className="flex items-center space-x-3">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center text-sm
                ${isSelected ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600'}
              `}>
                {actionIcons[index]}
              </div>
              
              <div className="flex-1">
                <div className="flex justify-between items-center mb-1">
                  <span className={`text-sm font-medium ${
                    isSelected ? 'text-blue-600' : 'text-gray-700'
                  }`}>
                    {actionNames[index]}
                  </span>
                  <span className="text-xs text-gray-500">
                    {qValue.toFixed(3)}
                  </span>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      isSelected ? 'bg-blue-500' : 'bg-gray-400'
                    }`}
                    style={{ width: `${normalizedValue * 100}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {selectedAction !== null && (
        <div className="mt-4 p-3 bg-blue-50 rounded-md">
          <div className="text-sm text-blue-800">
            <strong>선택된 액션:</strong> {actionNames[selectedAction]} {actionIcons[selectedAction]}
          </div>
        </div>
      )}
    </div>
  );
};