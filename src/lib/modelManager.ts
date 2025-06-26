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

  async predict(observation: Float32Array): Promise<ModelPrediction> {
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
      
      // 최고 Q-value를 가진 액션 선택
      const action = qValues.indexOf(Math.max(...qValues)) as GameAction;
      
      return {
        action,
        qValues
      };
    } catch (error) {
      console.error('❌ 모델 추론 실패:', error);
      console.error('입력 정보:', {
        observationLength: observation.length,
        expectedLength: 4 * 4 * 16,
        inputNames: this.session.inputNames,
        outputNames: this.session.outputNames
      });
      
      // 🔥 에러 발생 시 랜덤 액션 반환 (게임이 멈추지 않도록)
      const randomAction = Math.floor(Math.random() * 4) as GameAction;
      const randomQValues = Array.from({ length: 4 }, () => Math.random());
      
      console.warn(`⚠️ 랜덤 액션 사용: ${randomAction}`);
      
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