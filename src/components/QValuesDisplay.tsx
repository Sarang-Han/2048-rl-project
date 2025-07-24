import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faArrowUp, faArrowRight, faArrowDown, faArrowLeft, faRobot, faBan } from '@fortawesome/free-solid-svg-icons';
import { GameAction } from '@/types/game';
import { theme } from '@/lib/theme';

interface QValuesDisplayProps {
  qValues: number[] | null;
  selectedAction: GameAction | null;
  validActions?: GameAction[];
  className?: string;
}

export const QValuesDisplay: React.FC<QValuesDisplayProps> = ({ 
  qValues, 
  selectedAction, 
  validActions = [0, 1, 2, 3],
  className = '' 
}) => {
  const actionNames = ['Up', 'Right', 'Down', 'Left'];
  const actionIcons = [faArrowUp, faArrowRight, faArrowDown, faArrowLeft];

  if (!qValues) {
    return (
      <div className={`p-5 rounded-2xl flex flex-col ${className}`} style={{ background: theme.colors.controls.background }}>
        <h3 className="text-lg font-bold mb-4 text-center" style={{ color: theme.colors.primary.text }}>
          AI Predictions
        </h3>
        <div className="flex-1 flex items-center justify-center text-center p-4 rounded-xl border-2 border-dashed border-gray-300">
          <div style={{ color: theme.colors.primary.textSecondary }}>
            <FontAwesomeIcon icon={faRobot} className="text-3xl mb-3" />
            <div className="text-sm font-medium">Waiting for predictions...</div>
          </div>
        </div>
      </div>
    );
  }

  // 🔥 Q-values 정규화 (시각화 개선)
  const maxQ = Math.max(...qValues);
  const minQ = Math.min(...qValues);
  const range = maxQ - minQ;

  return (
    <div className={`p-5 rounded-2xl flex flex-col ${className}`} style={{ background: theme.colors.controls.background }}>
      <h3 className="text-lg font-bold mb-4 text-center" style={{ color: theme.colors.primary.text }}>
        AI Predictions
      </h3>
      
      {/* 🔥 액션 마스킹 상태 표시 */}
      <div className="mb-3 text-center">
        <div className="text-xs font-medium" style={{ color: theme.colors.primary.textSecondary }}>
          Valid Actions: {validActions.length}/4
        </div>
        <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
          <div 
            className="h-1 rounded-full transition-all duration-300"
            style={{ 
              width: `${(validActions.length / 4) * 100}%`,
              background: validActions.length === 4 ? theme.colors.status.success : 
                         validActions.length >= 2 ? theme.colors.status.warning : 
                         theme.colors.status.error
            }}
          />
        </div>
      </div>

      <div className="flex-1 flex flex-col space-y-3">
        {qValues.map((qValue, index) => {
          const isSelected = selectedAction === index;
          const isValid = validActions.includes(index as GameAction);
          const normalizedValue = range > 0 ? ((qValue - minQ) / range) * 100 : 50;

          return (
            <div key={index} className={`relative p-3 rounded-xl transition-all duration-300 ${
              isSelected ? 'ring-2 ring-blue-400' : ''
            }`} style={{
              background: isValid ? (isSelected ? '#e3f2fd' : '#ffffff') : '#f5f5f5',
              opacity: isValid ? 1 : 0.6
            }}>
              {/* 🔥 Q-value 바 배경 */}
              <div className="absolute inset-0 rounded-xl overflow-hidden">
                <div 
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${normalizedValue}%`,
                    background: isSelected ? 
                      'linear-gradient(90deg, rgba(59,130,246,0.1), rgba(59,130,246,0.2))' :
                      'linear-gradient(90deg, rgba(156,163,175,0.1), rgba(156,163,175,0.15))'
                  }}
                />
              </div>

              <div className="relative flex items-center space-x-3">
                <div className={`w-8 h-8 rounded-xl flex items-center justify-center text-sm font-bold relative ${
                  isSelected ? 'bg-blue-600 text-white' : 
                  isValid ? 'bg-gray-300 text-gray-700' : 'bg-gray-200 text-gray-400'
                }`}>
                  <FontAwesomeIcon icon={actionIcons[index]} />
                  {!isValid && (
                    <FontAwesomeIcon 
                      icon={faBan} 
                      className="absolute -top-1 -right-1 text-red-500 text-xs"
                    />
                  )}
                </div>
                
                <div className="flex-1">
                  <div className="flex justify-between items-center">
                    <span className={`text-sm font-bold ${
                      isSelected ? 'text-blue-800' : 
                      isValid ? 'text-gray-700' : 'text-gray-400'
                    }`}>
                      {actionNames[index]}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs font-mono font-bold px-2 py-1 rounded-lg ${
                        isSelected ? 'bg-blue-100 text-blue-800' : 'bg-gray-200 text-gray-700'
                      }`}>
                        {qValue.toFixed(3)}
                      </span>
                      {isSelected && <span className="text-xs text-blue-600">✓</span>}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};