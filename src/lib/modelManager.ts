import { InferenceSession, Tensor } from 'onnxruntime-web';
import { GameAction, ModelPrediction } from '@/types/game';

export class ModelManager {
  private session: InferenceSession | null = null;
  private isLoading: boolean = false;

  async loadModel(modelPath: string = '/models/cnn_model.onnx'): Promise<void> {
    if (this.isLoading) {
      console.warn('모델이 이미 로딩 중입니다.');
      return;
    }
    
    this.isLoading = true;
    try {
      this.session = await InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      console.log('✅ ONNX 모델 로드 완료');
      
    } catch (error) {
      console.error('❌ 모델 로드 실패:', error);
      this.session = null;
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  async predict(observation: Float32Array, validActions?: GameAction[]): Promise<ModelPrediction> {
    if (!this.session) {
      throw new Error('모델이 로드되지 않았습니다.');
    }

    try {
      const inputTensor = this.createInputTensor(observation);
      
      const inputName = this.session.inputNames[0] || 'state';
      const feeds = { [inputName]: inputTensor };
      const results = await this.session.run(feeds);
      
      const outputName = this.session.outputNames[0] || Object.keys(results)[0];
      const qValues = Array.from(results[outputName].data as Float32Array);
      
      // 🔥 Q-values 유효성 검사
      if (!qValues || qValues.length !== 4 || qValues.some(isNaN)) {
        throw new Error(`잘못된 Q-values: [${qValues.join(', ')}]`);
      }
      
      // 🔥 액션 마스킹 적용
      let selectedAction: GameAction;
      
      if (validActions && validActions.length > 0) {
        let bestAction = validActions[0];
        let bestQValue = qValues[validActions[0]];
        
        for (const action of validActions) {
          if (qValues[action] > bestQValue) {
            bestQValue = qValues[action];
            bestAction = action;
          }
        }
        
        selectedAction = bestAction;
        
      } else {
        selectedAction = qValues.indexOf(Math.max(...qValues)) as GameAction;
        console.warn('⚠️ 유효한 액션 정보가 없어 모든 액션 중에서 선택했습니다.');
      }
      
      return {
        action: selectedAction,
        qValues
      };
      
    } catch (error) {
      console.error('❌ 모델 추론 실패:', error);
      
      let randomAction: GameAction;
      if (validActions && validActions.length > 0) {
        randomAction = validActions[Math.floor(Math.random() * validActions.length)];
      } else {
        randomAction = Math.floor(Math.random() * 4) as GameAction;
      }
      
      return {
        action: randomAction,
        qValues: [0, 0, 0, 0]
      };
    }
  }

  private createInputTensor(observation: Float32Array): Tensor {
    if (observation.length !== 4 * 4 * 16) {
      throw new Error(`잘못된 관찰 데이터 크기: ${observation.length}, 예상: ${4 * 4 * 16}`);
    }
    
    return new Tensor('float32', observation, [1, 4, 4, 16]);
  }

  isReady(): boolean {
    return this.session !== null && !this.isLoading;
  }
  dispose(): void {
    if (this.session) {
      this.session = null;
      console.log('🗑️ 모델 세션이 정리되었습니다.');
    }
  }
}