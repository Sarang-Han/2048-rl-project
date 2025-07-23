# Solve 2048 by DQN

## Project

### 웹 애플리케이션 (Next.js)
- **Frontend**: Next.js + TypeScript + Tailwind CSS
- **AI 모델 실행**: ONNX Runtime Web으로 브라우저에서 직접 AI 모델 실행
- **실시간 게임 시각화**: AI의 플레이 과정과 Q-값 시각화

### AI 학습 (Python)
- **강화학습 알고리즘**: Deep Q-Network (DQN)
- **게임 환경**: 커스텀 2048 환경 구현
- **모델 아키텍처**: CNN 기반 네트워크
- **학습 완료 모델**: ONNX 형식으로 웹에서 사용 가능

## How to run

### 웹 애플리케이션 실행
```bash
npm install
npm run dev
```

### AI 모델 학습 (선택사항)
```bash
cd training
pip install -r requirements.txt
# 학습 코드 실행 (Jupyter Notebook 참고)
```

## TODO

- [x] 2048 환경, 환경 테스트 추가
- [x] 간단한 DQN 알고리즘 코드, 통합 테스트 추가
- [x] `__init__` 추가
- [x] 2048 1차 GUI 추가 
- [x] 2048 GUI 디자인 개선 - [참고](https://github.com/gabrielecirulli/2048)
- [x] Illegal move 예외처리 관련
- [x] DQN 알고리즘 개선
- [ ] 액션 마스킹 적용 관련