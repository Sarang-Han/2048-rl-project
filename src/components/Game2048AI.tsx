'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Game2048 } from '@/lib/game2048';
import { ModelManager } from '@/lib/modelManager';
import { GameBoard } from './GameBoard';
import { GameInfo } from './GameInfo';
import { GameControls } from './GameControls';
import { QValuesDisplay } from './QValuesDisplay';
import { GameState, GameSpeed, ModelPrediction } from '@/types/game';

export const Game2048AI: React.FC = () => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [speed, setSpeed] = useState<GameSpeed>(1);
  const [currentPrediction, setCurrentPrediction] = useState<ModelPrediction | null>(null);
  const [errorCount, setErrorCount] = useState(0);

  const gameRef = useRef<Game2048>();
  const modelRef = useRef<ModelManager>();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // 게임 및 모델 초기화
  useEffect(() => {
    const initializeGame = async () => {
      try {
        // 게임 초기화
        gameRef.current = new Game2048();
        setGameState(gameRef.current.getState());

        // 모델 초기화
        modelRef.current = new ModelManager();
        await modelRef.current.loadModel('/models/cnn_model.onnx');
        setIsModelReady(true);
        
        console.log('✅ 게임 및 모델 초기화 완료');
      } catch (error) {
        console.error('❌ 초기화 실패:', error);
        setIsModelReady(false);
      }
    };

    initializeGame();

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // 게임 컨트롤 함수들 - handleReset을 먼저 정의
  const handleReset = useCallback(() => {
    if (gameRef.current) {
      setIsPlaying(false);
      setErrorCount(0);
      const newState = gameRef.current.reset();
      setGameState(newState);
      setCurrentPrediction(null);
    }
  }, []);

  // 🔥 개선된 게임 스텝 실행 (에러 처리 강화)
  const executeGameStep = useCallback(async () => {
    if (!gameRef.current || !modelRef.current || !isModelReady) {
      return;
    }

    try {
      // 게임이 이미 종료되었는지 확인
      if (gameState?.gameOver) {
        setIsPlaying(false);
        return;
      }

      const observation = gameRef.current.getObservation();
      console.log('관찰 데이터 길이:', observation.length); // 디버깅용
      
      const prediction = await modelRef.current.predict(observation);
      setCurrentPrediction(prediction);
      
      // 🔥 유효한 액션인지 미리 확인
      const validActions = gameRef.current.getValidActions();
      let selectedAction = prediction.action;
      
      // 예측된 액션이 유효하지 않으면 유효한 액션 중 하나 선택
      if (validActions.length > 0 && !validActions.includes(prediction.action)) {
        selectedAction = validActions[0];
        console.warn(`⚠️ 유효하지 않은 액션 ${prediction.action}, ${selectedAction}로 변경`);
      }
      
      const result = gameRef.current.step(selectedAction);
      setGameState(result.state);
    
      
      // 에러 카운트 리셋
      setErrorCount(0);
      
      // 게임 종료 확인
      if (result.done) {
        setIsPlaying(false);
        console.log(`🏁 게임 종료! 최종 점수: ${result.state.score}, 최고 타일: ${result.state.highest}`);
      }
      
    } catch (error) {
      console.error('❌ 게임 스텝 실행 실패:', error);
      
      // 🔥 에러 발생 시 카운트 증가 및 처리
      setErrorCount(prev => {
        const newCount = prev + 1;
        
        // 연속 에러가 5회 이상이면 게임 중단
        if (newCount >= 5) {
          console.error('❌ 연속 에러 5회 발생, 게임 중단');
          setIsPlaying(false);
          
          // 게임 리셋 제안
          setTimeout(() => {
            if (confirm('게임에서 오류가 발생했습니다. 새 게임을 시작하시겠습니까?')) {
              handleReset();
            }
          }, 1000);
        }
        
        return newCount;
      });
    }
  }, [isModelReady, gameState?.gameOver, handleReset]);

  const handlePlay = useCallback(() => {
    if (gameState?.gameOver) {
      handleReset();
    }
    setErrorCount(0); // 플레이 시작 시 에러 카운트 리셋
    setIsPlaying(true);
  }, [gameState?.gameOver, handleReset]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const handleSpeedChange = useCallback((newSpeed: GameSpeed) => {
    setSpeed(newSpeed);
  }, []);

  // 게임 루프
  useEffect(() => {
    if (isPlaying && isModelReady) {
      const delay = 1000 / speed;
      intervalRef.current = setInterval(executeGameStep, delay);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, speed, executeGameStep, isModelReady]);

  if (!gameState) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">게임을 초기화하는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="container mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            2048 AI Demo
          </h1>
          {/* 🔥 에러 상태 표시 */}
          {errorCount > 0 && (
            <div className="mt-2 text-orange-600 text-sm">
              ⚠️ 오류 발생 횟수: {errorCount}/5
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 게임 보드 */}
          <div className="lg:col-span-2 space-y-6">
            <div className="flex justify-center">
              <GameBoard gameState={gameState} />
            </div>
            
            <GameInfo 
              gameState={gameState}
              isPlaying={isPlaying}
              speed={speed}
            />
          </div>

          {/* 사이드 패널 */}
          <div className="space-y-6">
            <GameControls
              isPlaying={isPlaying}
              isModelReady={isModelReady}
              speed={speed}
              onPlay={handlePlay}
              onPause={handlePause}
              onReset={handleReset}
              onSpeedChange={handleSpeedChange}
            />

            <QValuesDisplay
              qValues={currentPrediction?.qValues || null}
              selectedAction={currentPrediction?.action || null}
            />
          </div>
        </div>
      </div>
    </div>
  );
};