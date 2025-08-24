'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Game2048 } from '@/lib/game2048';
import { ModelManager } from '@/lib/modelManager';
import { GameBoard } from './GameBoard';
import { GameInfo } from './GameInfo';
import { GameControls } from './GameControls';
import { QValuesDisplay } from './QValuesDisplay';
import { GameState, GameSpeed, ModelPrediction, GameAction } from '@/types/game';

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

  const gameRef = useRef<Game2048 | undefined>(undefined);
  const modelRef = useRef<ModelManager | undefined>(undefined);
  const isUnmountedRef = useRef(false);

  useEffect(() => {
    return () => {
      isUnmountedRef.current = true;
    };
  }, []);

  const handleReset = useCallback(() => {
    if (gameRef.current && !isUnmountedRef.current) {
      setIsPlaying(false);
      setErrorCount(0);
      const newState = gameRef.current.reset();
      setGameState(newState);
      setCurrentPrediction(null);
      setCurrentValidActions(gameRef.current.getValidActions());
    }
  }, []);

  // 게임 및 모델 초기화
  useEffect(() => {
    const initializeGame = async () => {
      try {
        if (isUnmountedRef.current) return;
        
        gameRef.current = new Game2048();
        setGameState(gameRef.current.getState());
        setCurrentValidActions(gameRef.current.getValidActions());

        modelRef.current = new ModelManager();
        await modelRef.current.loadModel('/models/cnn_model.onnx');
        
        if (!isUnmountedRef.current) {
          setIsModelReady(true);
          console.log('✅ 게임 및 모델 초기화 완료');
        }
      } catch (error) {
        if (!isUnmountedRef.current) {
          console.error('❌ 초기화 실패:', error);
          setIsModelReady(false);
        }
      }
    };

    initializeGame();
  }, []);

  const executeGameStep = useCallback(async () => {
    if (!gameRef.current || !modelRef.current || !isModelReady || 
        gameState?.gameOver || isUnmountedRef.current) {
      if (gameState?.gameOver) setIsPlaying(false);
      return;
    }

    try {
      const validActions = gameRef.current.getValidActions();
      setCurrentValidActions(validActions);
      
      if (validActions.length === 0) {
        console.log('🏁 게임 종료 - 유효한 액션 없음');
        setIsPlaying(false);
        return;
      }

      const observation = gameRef.current.getObservation();
      const prediction = await modelRef.current.predict(observation, validActions);
      
      if (isUnmountedRef.current) return;
      
      setCurrentPrediction(prediction);
      
      if (!validActions.includes(prediction.action)) {
        console.warn(`⚠️ 잘못된 액션 ${prediction.action}, 대체 액션: ${validActions[0]}`);
        prediction.action = validActions[0];
      }
      
      const result = gameRef.current.step(prediction.action);
      setGameState(result.state);
      setErrorCount(0);
      
      if (result.done) {
        setIsPlaying(false);
        console.log(`🏁 게임 완료! 점수: ${result.state.score.toLocaleString()}, 최고: ${result.state.highest}`);
        
        // 🔥 통계 업데이트
        setGameStats(prev => ({
          totalGames: prev.totalGames + 1,
          bestScore: Math.max(prev.bestScore, result.state.score),
          averageScore: (prev.averageScore * prev.totalGames + result.state.score) / (prev.totalGames + 1),
          gamesWon: result.state.highest >= 2048 ? prev.gamesWon + 1 : prev.gamesWon
        }));
      }
      
    } catch (error) {
      if (isUnmountedRef.current) return;
      
      console.error('❌ 게임 스텝 실행 실패:', error);
      setErrorCount(prev => {
        const newCount = prev + 1;
        if (newCount >= 3) {
          setIsPlaying(false);
          setTimeout(() => {
            if (!isUnmountedRef.current && 
                confirm('연속 오류가 발생했습니다. 새 게임을 시작하시겠습니까?')) {
              handleReset();
            }
          }, 1000);
        }
        return newCount;
      });
    }
  }, [isModelReady, gameState?.gameOver, handleReset]);

  const handlePlay = useCallback(() => {
    if (gameState?.gameOver) handleReset();
    setErrorCount(0);
    setIsPlaying(true);
  }, [gameState?.gameOver, handleReset]);

  const handlePause = useCallback(() => setIsPlaying(false), []);
  const handleSpeedChange = useCallback((newSpeed: GameSpeed) => setSpeed(newSpeed), []);

  useEffect(() => {
    if (!isPlaying || !isModelReady) return;

    const stepInterval = 300 / speed;
    let lastStepTime = 0;
    let animationFrameId: number;

    const gameLoop = (timestamp: number) => {
      if (isUnmountedRef.current) return;
      
      if (timestamp - lastStepTime >= stepInterval) {
        executeGameStep();
        lastStepTime = timestamp;
      }
      
      if (isPlaying && isModelReady && !isUnmountedRef.current) {
        animationFrameId = requestAnimationFrame(gameLoop);
      }
    };

    animationFrameId = requestAnimationFrame(gameLoop);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isPlaying, isModelReady, speed, executeGameStep]);

  if (!gameState) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: '#faf8ef' }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 mx-auto mb-4" style={{ borderColor: '#8f7a66' }}></div>
          <p className="text-lg font-medium" style={{ color: '#776e65' }}>AI 시스템 초기화 중...</p>
          <p className="text-sm mt-2" style={{ color: '#8f7a66' }}>모델을 로드하고 있습니다</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen overflow-auto flex flex-col" style={{ backgroundColor: '#faf8ef' }}>
      {/* 헤더 */}
      <header className="flex-shrink-0 text-center py-8 px-4">
        <h1 className="text-5xl font-black mb-2" style={{ color: '#776e65' }}>2048 AI Demo</h1>
        <p className="text-base font-medium max-w-2xl mx-auto" style={{ color: '#8f7a66' }}>
          Deep Reinforcement Learning with Action Masking
        </p>
        
        {gameStats.totalGames > 0 && (
          <div className="mt-3 flex justify-center items-center space-x-6 text-sm" style={{ color: '#8f7a66' }}>
            <span>Games: <strong>{gameStats.totalGames}</strong></span>
            <span>Best: <strong>{gameStats.bestScore.toLocaleString()}</strong></span>
            <span>Avg: <strong>{Math.round(gameStats.averageScore).toLocaleString()}</strong></span>
            <span>Won: <strong>{gameStats.gamesWon}</strong></span>
          </div>
        )}
      </header>

      {/* 메인 콘텐츠 */}
      <main className="flex-1 px-4 pb-4 min-h-0">
        <ResponsiveLayout
          gameState={gameState}
          errorCount={errorCount}
          isPlaying={isPlaying}
          isModelReady={isModelReady}
          speed={speed}
          currentPrediction={currentPrediction}
          currentValidActions={currentValidActions}
          onPlay={handlePlay}
          onPause={handlePause}
          onReset={handleReset}
          onSpeedChange={handleSpeedChange}
        />
      </main>
    </div>
  );
};

interface ResponsiveLayoutProps {
  gameState: GameState;
  errorCount: number;
  isPlaying: boolean;
  isModelReady: boolean;
  speed: GameSpeed;
  currentPrediction: ModelPrediction | null;
  currentValidActions: GameAction[];
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onSpeedChange: (speed: GameSpeed) => void;
}

const ResponsiveLayout: React.FC<ResponsiveLayoutProps> = ({
  gameState, errorCount, isPlaying, isModelReady, speed,
  currentPrediction, currentValidActions,
  onPlay, onPause, onReset, onSpeedChange
}) => (
  <>
    {/* 데스크톱 레이아웃 */}
    <div className="hidden xl:flex items-center justify-center h-full">
      <div className="flex gap-6 items-start">
        <div className="flex-shrink-0" style={{ width: '280px' }}>
          <GameControls
            isPlaying={isPlaying} isModelReady={isModelReady} speed={speed}
            onPlay={onPlay} onPause={onPause} onReset={onReset} onSpeedChange={onSpeedChange}
            className="h-full"
          />
        </div>

        <div className="flex flex-col items-center flex-shrink-0">
          <div className="mb-4"><GameBoard gameState={gameState} /></div>
          <div style={{ width: '440px' }}><GameInfo gameState={gameState} errorCount={errorCount} /></div>
        </div>

        <div className="flex-shrink-0" style={{ width: '300px' }}>
          <QValuesDisplay
            qValues={currentPrediction?.qValues || null}
            selectedAction={currentPrediction?.action || null}
            validActions={currentValidActions}
            className="h-full"
          />
        </div>
      </div>
    </div>

    {/* 태블릿 레이아웃 */}
    <div className="hidden lg:flex xl:hidden flex-col items-center justify-center h-full space-y-6">
      <div className="flex flex-col items-center">
        <div className="mb-4"><GameBoard gameState={gameState} /></div>
        <div style={{ width: '440px' }}><GameInfo gameState={gameState} errorCount={errorCount} /></div>
      </div>

      <div className="flex gap-6 justify-center">
        <div style={{ width: '280px' }}>
          <GameControls
            isPlaying={isPlaying} isModelReady={isModelReady} speed={speed}
            onPlay={onPlay} onPause={onPause} onReset={onReset} onSpeedChange={onSpeedChange}
          />
        </div>
        <div style={{ width: '300px' }}>
          <QValuesDisplay
            qValues={currentPrediction?.qValues || null}
            selectedAction={currentPrediction?.action || null}
            validActions={currentValidActions}
          />
        </div>
      </div>
    </div>

    {/* 모바일 레이아웃 */}
    <div className="lg:hidden flex flex-col space-y-4 max-w-md mx-auto">
      <div className="flex flex-col items-center">
        <div className="mb-4 w-full flex justify-center">
          <div className="transform scale-75"><GameBoard gameState={gameState} /></div>
        </div>
        <div className="w-full"><GameInfo gameState={gameState} errorCount={errorCount} /></div>
      </div>

      <div className="space-y-4">
        <GameControls
          isPlaying={isPlaying} isModelReady={isModelReady} speed={speed}
          onPlay={onPlay} onPause={onPause} onReset={onReset} onSpeedChange={onSpeedChange}
          className="w-full"
        />
        <QValuesDisplay
          qValues={currentPrediction?.qValues || null}
          selectedAction={currentPrediction?.action || null}
          validActions={currentValidActions}
          className="w-full"
        />
      </div>
    </div>
  </>
);