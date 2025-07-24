'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Game2048 } from '@/lib/game2048';
import { ModelManager } from '@/lib/modelManager';
import { GameBoard } from './GameBoard';
import { GameInfo } from './GameInfo';
import { GameControls } from './GameControls';
import { QValuesDisplay } from './QValuesDisplay';
import { GameState, GameSpeed, ModelPrediction, GameAction } from '@/types/game'; // 🔥 GameAction 추가

export const Game2048AI: React.FC = () => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [speed, setSpeed] = useState<GameSpeed>(1);
  const [currentPrediction, setCurrentPrediction] = useState<ModelPrediction | null>(null);
  const [errorCount, setErrorCount] = useState(0);
  const [currentValidActions, setCurrentValidActions] = useState<GameAction[]>([0, 1, 2, 3]);
  const [gameStats, setGameStats] = useState({
    totalGames: 0,
    bestScore: 0,
    averageScore: 0,
    gamesWon: 0
  });

  const gameRef = useRef<Game2048>();
  const modelRef = useRef<ModelManager>();

  // 게임 컨트롤 함수들 - handleReset을 먼저 정의
  const handleReset = useCallback(() => {
    if (gameRef.current) {
      setIsPlaying(false);
      setErrorCount(0);
      const newState = gameRef.current.reset();
      setGameState(newState);
      setCurrentPrediction(null);
      
      const validActions = gameRef.current.getValidActions();
      setCurrentValidActions(validActions);
    }
  }, []);

  // 게임 및 모델 초기화
  useEffect(() => {
    const initializeGame = async () => {
      try {
        // 게임 초기화
        gameRef.current = new Game2048();
        setGameState(gameRef.current.getState());
        
        // 🔥 초기 유효한 액션들 설정
        const initialValidActions = gameRef.current.getValidActions();
        setCurrentValidActions(initialValidActions);

        // 모델 초기화
        modelRef.current = new ModelManager();
        await modelRef.current.loadModel('/models/cnn_model.onnx');
        setIsModelReady(true);
        
        console.log('✅ 게임 및 모델 초기화 완료 (액션 마스킹 지원)');
      } catch (error) {
        console.error('❌ 초기화 실패:', error);
        setIsModelReady(false);
      }
    };

    initializeGame();
  }, []);

  // 🔥 액션 마스킹이 완전히 적용된 게임 스텝 실행
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

      // 🔥 1단계: 유효한 액션들 확인
      const validActions = gameRef.current.getValidActions();
      setCurrentValidActions(validActions); // 🔥 상태 업데이트 추가
      
      if (validActions.length === 0) {
        console.log('🏁 게임 종료 - 유효한 액션이 없습니다.');
        setIsPlaying(false);
        return;
      }

      // 🔥 2단계: 관찰 데이터 생성
      const observation = gameRef.current.getObservation();
      
      // 🔥 3단계: 액션 마스킹이 적용된 모델 예측
      const prediction = await modelRef.current.predict(observation, validActions);
      setCurrentPrediction(prediction);
      
      // 🔥 4단계: 선택된 액션이 유효한지 재확인 (안전장치)
      if (!validActions.includes(prediction.action)) {
        console.error(`❌ 심각한 오류: 모델이 유효하지 않은 액션(${prediction.action})을 선택했습니다!`);
        console.error(`   유효한 액션들: [${validActions.join(', ')}]`);
        
        // 강제로 첫 번째 유효한 액션 사용
        prediction.action = validActions[0];
        console.warn(`   -> 강제로 ${prediction.action} 액션 사용`);
      }
      
      // 🔥 5단계: 액션 실행
      const result = gameRef.current.step(prediction.action);
      setGameState(result.state);
      
      // 성공적인 스텝 - 에러 카운트 리셋
      setErrorCount(0);
      
      // 게임 종료 확인
      if (result.done) {
        setIsPlaying(false);
        console.log(`🏁 게임 종료!`);
        console.log(`   최종 점수: ${result.state.score.toLocaleString()}`);
        console.log(`   최고 타일: ${result.state.highest}`);
        console.log(`   총 스텝: ${result.state.steps}`);

        setGameStats(prevStats => {
          const newTotalGames = prevStats.totalGames + 1;
          const newBestScore = Math.max(prevStats.bestScore, result.state.score);
          const newAverageScore = (prevStats.averageScore * prevStats.totalGames + result.state.score) / newTotalGames;
          const newGamesWon = result.state.highest === 2048 ? prevStats.gamesWon + 1 : prevStats.gamesWon; // 2048 달성 시 승리 게임 수 증가
          
          return {
            totalGames: newTotalGames,
            bestScore: newBestScore,
            averageScore: newAverageScore,
            gamesWon: newGamesWon
          };
        });
      } else {
        if (result.state.steps % 10 === 0) {
          console.log(`📊 Step ${result.state.steps}: Score=${result.state.score}, Highest=${result.state.highest}, Valid=${validActions.length}`);
        }
      }
      
    } catch (error) {
      console.error('❌ 게임 스텝 실행 실패:', error);
      
      // 🔥 에러 발생 시 카운트 증가 및 처리
      setErrorCount(prev => {
        const newCount = prev + 1;
        
        // 연속 에러가 3회 이상이면 게임 중단 (액션 마스킹으로 더 엄격하게)
        if (newCount >= 3) {
          console.error('❌ 연속 에러 3회 발생, 게임 중단');
          setIsPlaying(false);
          
          // 게임 리셋 제안
          setTimeout(() => {
            if (confirm('액션 마스킹에도 불구하고 오류가 발생했습니다. 새 게임을 시작하시겠습니까?')) {
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
    let timeoutId: NodeJS.Timeout | null = null;
    let lastStepTime = 0;

    const gameLoop = async () => {
      if (!isPlaying || !isModelReady) return;

      const now = Date.now();
      const stepInterval = 500 / speed; // speed에 따른 인터벌 계산
      
      if (now - lastStepTime >= stepInterval) {
        await executeGameStep();
        lastStepTime = now;
      }
      
      // 다음 루프 스케줄링
      if (isPlaying && isModelReady) {
        timeoutId = setTimeout(gameLoop, Math.min(50, stepInterval / 4)); // 최대 20fps
      }
    };

    if (isPlaying && isModelReady) {
      gameLoop();
    }

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [isPlaying, isModelReady, speed, executeGameStep]);


  if (!gameState) {
    return (
      <div 
        className="min-h-screen flex items-center justify-center"
        style={{ background: '#faf8ef' }}
      >
        <div className="text-center">
          <div 
            className="animate-spin rounded-full h-12 w-12 border-b-2 mx-auto"
            style={{ borderColor: '#8f7a66' }}
          ></div>
          <p className="mt-4" style={{ color: '#776e65' }}>액션 마스킹 AI 시스템 초기화 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="min-h-screen overflow-auto flex flex-col"
      style={{ backgroundColor: '#faf8ef' }}
    >
      {/* 헤더 */}
      <div className="flex-shrink-0 text-center py-8 px-4">
        <p
          className="text-5xl font-black mb-2"
          style={{ color: '#776e65' }}
        >
          2048 AI Demo
        </p>
        <p 
          className="text-base font-medium leading-relaxed max-w-2xl mx-auto"
          style={{ color: '#8f7a66' }}
        >
          Deep Reinforcement Learning with Action Masking
        </p>
        
        {/* 게임 통계 표시 */}
        {gameStats.totalGames > 0 && (
          <div className="mt-2 flex justify-center items-center space-x-6 text-xs">
            <div style={{ color: '#8f7a66' }}>
              Games: {gameStats.totalGames}
            </div>
            <div style={{ color: '#8f7a66' }}>
              Best: {gameStats.bestScore.toLocaleString()}
            </div>
            <div style={{ color: '#8f7a66' }}>
              Avg: {Math.round(gameStats.averageScore).toLocaleString()}
            </div>
            <div style={{ color: '#8f7a66' }}>
              Won: {gameStats.gamesWon}
            </div>
          </div>
        )}
      </div>

      {/* 메인 콘텐츠 - 기존과 동일 */}
      <div className="flex-1 px-4 pb-4 min-h-0">
        {/* 데스크톱 레이아웃 - 최소 너비 제한 */}
        <div className="hidden xl:flex items-center justify-center h-full">
          <div className="flex gap-6 items-start">
            {/* 게임 컨트롤 */}
            <div className="flex-shrink-0" style={{ width: '280px' }}>
              <GameControls
                isPlaying={isPlaying}
                isModelReady={isModelReady}
                speed={speed}
                onPlay={handlePlay}
                onPause={handlePause}
                onReset={handleReset}
                onSpeedChange={handleSpeedChange}
                className="h-full"
              />
            </div>

            {/* 게임 영역 (게임보드 + 정보) - 고정 크기 */}
            <div className="flex flex-col items-center flex-shrink-0">
              <div className="mb-4">
                <GameBoard gameState={gameState} />
              </div>
              <div style={{ width: '440px' }}>
                <GameInfo 
                  gameState={gameState}
                  errorCount={errorCount}
                />
              </div>
            </div>

            {/* 🔥 Q값 디스플레이 - validActions 전달 추가 */}
            <div className="flex-shrink-0" style={{ width: '300px' }}>
              <QValuesDisplay
                qValues={currentPrediction?.qValues || null}
                selectedAction={currentPrediction?.action || null}
                validActions={currentValidActions} // 🔥 추가
                className="h-full"
              />
            </div>
          </div>
        </div>

        {/* 중간 크기 화면 레이아웃 (태블릿) */}
        <div className="hidden lg:flex xl:hidden flex-col items-center justify-center h-full space-y-6">
          {/* 게임 영역 상단 */}
          <div className="flex flex-col items-center">
            <div className="mb-4">
              <GameBoard gameState={gameState} />
            </div>
            <div style={{ width: '440px' }}>
              <GameInfo 
                gameState={gameState}
                errorCount={errorCount}
              />
            </div>
          </div>

          {/* 컨트롤 패널들 하단 */}
          <div className="flex gap-6 justify-center">
            <div style={{ width: '280px' }}>
              <GameControls
                isPlaying={isPlaying}
                isModelReady={isModelReady}
                speed={speed}
                onPlay={handlePlay}
                onPause={handlePause}
                onReset={handleReset}
                onSpeedChange={handleSpeedChange}
              />
            </div>
            {/* 🔥 validActions 전달 추가 */}
            <div style={{ width: '300px' }}>
              <QValuesDisplay
                qValues={currentPrediction?.qValues || null}
                selectedAction={currentPrediction?.action || null}
                validActions={currentValidActions} // 🔥 추가
              />
            </div>
          </div>
        </div>

        {/* 모바일/소형 태블릿 레이아웃 */}
        <div className="lg:hidden flex flex-col space-y-4 max-w-md mx-auto">
          {/* 게임 영역 (게임보드 + 정보) */}
          <div className="flex flex-col items-center">
            <div className="mb-4 w-full flex justify-center">
              <div className="transform scale-75">
                <GameBoard gameState={gameState} />
              </div>
            </div>
            <div className="w-full">
              <GameInfo 
                gameState={gameState}
                errorCount={errorCount}
              />
            </div>
          </div>

          {/* 컨트롤 패널들 - 세로 배치 */}
          <div className="space-y-4">
            <div>
              <GameControls
                isPlaying={isPlaying}
                isModelReady={isModelReady}
                speed={speed}
                onPlay={handlePlay}
                onPause={handlePause}
                onReset={handleReset}
                onSpeedChange={handleSpeedChange}
                className="w-full"
              />
            </div>

            {/* 🔥 validActions 전달 추가 */}
            <div>
              <QValuesDisplay
                qValues={currentPrediction?.qValues || null}
                selectedAction={currentPrediction?.action || null}
                validActions={currentValidActions} // 🔥 추가
                className="w-full"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};