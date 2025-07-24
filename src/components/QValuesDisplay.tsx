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
      <div 
        className={`p-5 rounded-xl flex flex-col ${className}`} 
        style={{ 
          backgroundColor: theme.colors.controls.background,
          border: `${theme.borders.medium} ${theme.colors.controls.border}`
        }}
      >
        <h3 
          className="text-lg font-bold mb-4 text-center" 
          style={{ color: theme.colors.primary.text }}
        >
          AI Predictions
        </h3>
        <div 
          className="flex-1 flex items-center justify-center text-center p-4 rounded-lg"
          style={{
            border: `${theme.borders.thin} ${theme.colors.controls.border}`,
            borderStyle: 'dashed'
          }}
        >
          <div style={{ color: theme.colors.primary.textSecondary }}>
            <FontAwesomeIcon icon={faRobot} className="text-3xl mb-3" />
            <div className="text-sm font-medium">Waiting for predictions...</div>
          </div>
        </div>
      </div>
    );
  }

  // Q-values 정규화 (시각화 개선)
  const maxQ = Math.max(...qValues);
  const minQ = Math.min(...qValues);
  const range = maxQ - minQ;

  return (
    <div 
      className={`p-5 rounded-xl flex flex-col ${className}`} 
      style={{ 
        backgroundColor: theme.colors.controls.background,
        border: `${theme.borders.medium} ${theme.colors.controls.border}`
      }}
    >
      <h3 
        className="text-lg font-bold mb-4 text-center" 
        style={{ color: theme.colors.primary.text }}
      >
        AI Predictions
      </h3>
      
      {/* 액션 마스킹 상태 표시*/}
      <div className="mb-3 text-center">
        <div 
          className="text-xs font-medium" 
          style={{ color: theme.colors.primary.textSecondary }}
        >
          Valid Actions: {validActions.length}/4
        </div>
        <div 
          className="w-full rounded-full h-2 mt-1"
          style={{ 
            backgroundColor: theme.colors.board.cellEmpty,
            border: `1px solid ${theme.colors.controls.border}`
          }}
        >
          <div 
            className="h-full rounded-full transition-all duration-300"
            style={{ 
              width: `${(validActions.length / 4) * 100}%`,
              backgroundColor: theme.colors.controls.button.primary
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
            <div 
              key={index} 
              className="relative p-3 rounded-lg transition-all duration-300"
              style={{
                backgroundColor: isValid ? 
                  (isSelected ? theme.colors.primary.background : theme.colors.controls.background) : 
                  theme.colors.board.cellEmpty,
                border: `${theme.borders.thin} ${isSelected ? 
                  theme.colors.controls.button.primary : 
                  theme.colors.controls.border}`,
                opacity: isValid ? 1 : 0.5
              }}
            >
              {/* Q-value 바 배경 - 통일된 색상으로 변경 */}
              <div className="absolute inset-0 rounded-lg overflow-hidden">
                <div 
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${normalizedValue}%`,
                    backgroundColor: isSelected ? 
                      'rgba(143, 122, 102, 0.1)' :  // theme.colors.controls.button.primary의 투명 버전
                      'rgba(187, 173, 160, 0.1)'    // theme.colors.controls.border의 투명 버전
                  }}
                />
              </div>

              <div className="relative flex items-center space-x-3">
                <div 
                  className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold relative"
                  style={{
                    backgroundColor: isSelected ? 
                      theme.colors.controls.button.primary : 
                      (isValid ? theme.colors.controls.button.secondary : theme.colors.board.cellEmpty),
                    color: isSelected ? 'white' : theme.colors.primary.text,
                    border: `${theme.borders.thin} ${isSelected ? 
                      theme.colors.controls.button.primaryHover : 
                      theme.colors.controls.border}`
                  }}
                >
                  <FontAwesomeIcon icon={actionIcons[index]} />
                  {!isValid && (
                    <FontAwesomeIcon 
                      icon={faBan} 
                      className="absolute -top-1 -right-1 text-xs"
                      style={{ color: theme.colors.primary.textSecondary }}
                    />
                  )}
                </div>
                
                <div className="flex-1">
                  <div className="flex justify-between items-center">
                    <span 
                      className="text-sm font-bold"
                      style={{
                        color: isSelected ? 
                          theme.colors.controls.button.primary : 
                          (isValid ? theme.colors.primary.text : theme.colors.primary.textSecondary)
                      }}
                    >
                      {actionNames[index]}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span 
                        className="text-xs font-mono font-bold px-2 py-1 rounded-md"
                        style={{
                          backgroundColor: isSelected ? 
                            theme.colors.primary.background : 
                            theme.colors.board.cellEmpty,
                          color: isSelected ? 
                            theme.colors.controls.button.primary : 
                            theme.colors.primary.text,
                          border: `1px solid ${isSelected ? 
                            theme.colors.controls.button.primary : 
                            theme.colors.controls.border}`
                        }}
                      >
                        {qValue.toFixed(3)}
                      </span>
                      {isSelected && (
                        <span style={{ color: theme.colors.controls.button.primary }}>✓</span>
                      )}
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