import { InferenceSession, Tensor } from 'onnxruntime-web';
import { GameAction, ModelPrediction } from '@/types/game';

export class ModelManager {
  private session: InferenceSession | null = null;
  private isLoading: boolean = false;

  async loadModel(modelPath: string = '/models/cnn_model.onnx'): Promise<void> {
    if (this.isLoading) return;
    
    this.isLoading = true;
    try {
      this.session = await InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      console.log('✅ ONNX 모델 로드 완료');
      
      // 모델 입출력 정보 출력
      console.log('📊 모델 정보:');
      console.log('  입력:', this.session.inputNames);
      console.log('  출력:', this.session.outputNames);
      
    } catch (error) {
      console.error('❌ 모델 로드 실패:', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  // 🔥 액션 마스킹이 적용된 예측 함수
  async predict(observation: Float32Array, validActions?: GameAction[]): Promise<ModelPrediction> {
    if (!this.session) {
      throw new Error('모델이 로드되지 않았습니다.');
    }

    try {
      // 입력 텐서 생성 - (4, 4, 16) 형태를 (1, 4, 4, 16)으로 변환
      const inputTensor = this.createInputTensor(observation);
      
      // 추론 실행
      const inputName = this.session.inputNames[0] || 'state';
      const feeds = { [inputName]: inputTensor };
      const results = await this.session.run(feeds);
      
      // Q-values 추출
      const outputName = this.session.outputNames[0] || Object.keys(results)[0];
      const qValues = Array.from(results[outputName].data as Float32Array);
      
      // 🔥 액션 마스킹 적용
      let selectedAction: GameAction;
      
      if (validActions && validActions.length > 0) {
        // 유효한 액션 중에서만 선택
        let bestAction = validActions[0];
        let bestQValue = qValues[validActions[0]];
        
        for (const action of validActions) {
          if (qValues[action] > bestQValue) {
            bestQValue = qValues[action];
            bestAction = action;
          }
        }
        
        selectedAction = bestAction;
        
        // 디버깅 정보
        console.log('🎭 액션 마스킹 적용:');
        console.log(`  유효한 액션들: [${validActions.join(', ')}]`);
        console.log(`  각 Q-values: [${validActions.map(a => `${a}:${qValues[a].toFixed(3)}`).join(', ')}]`);
        console.log(`  선택된 액션: ${selectedAction} (Q=${qValues[selectedAction].toFixed(3)})`);
        
      } else {
        // 모든 액션 중에서 최고 Q-value 선택 (백업)
        selectedAction = qValues.indexOf(Math.max(...qValues)) as GameAction;
        console.warn('⚠️ 유효한 액션 정보가 없어 모든 액션 중에서 선택했습니다.');
      }
      
      return {
        action: selectedAction,
        qValues
      };
      
    } catch (error) {
      console.error('❌ 모델 추론 실패:', error);
      console.error('입력 정보:', {
        observationLength: observation.length,
        expectedLength: 4 * 4 * 16,
        validActions: validActions,
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames
      });
      
      // 🔥 에러 발생 시 유효한 액션 중에서 랜덤 선택
      let randomAction: GameAction;
      if (validActions && validActions.length > 0) {
        randomAction = validActions[Math.floor(Math.random() * validActions.length)];
        console.warn(`⚠️ 유효한 액션 중 랜덤 선택: ${randomAction}`);
      } else {
        randomAction = Math.floor(Math.random() * 4) as GameAction;
        console.warn(`⚠️ 전체 액션 중 랜덤 선택: ${randomAction}`);
      }
      
      const randomQValues = Array.from({ length: 4 }, () => Math.random());
      
      return {
        action: randomAction,
        qValues: randomQValues
      };
    }
  }

  private createInputTensor(observation: Float32Array): Tensor {
    if (observation.length !== 4 * 4 * 16) {
      throw new Error(`잘못된 관찰 데이터 크기: ${observation.length}, 예상: ${4 * 4 * 16}`);
    }
    
    // (4, 4, 16) 데이터를 (1, 4, 4, 16) 배치 형태로 변환
    return new Tensor('float32', observation, [1, 4, 4, 16]);
  }

  isReady(): boolean {
    return this.session !== null && !this.isLoading;
  }
}