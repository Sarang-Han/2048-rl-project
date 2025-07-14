import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.game_2048 import Game2048Env
from models.dqn_agent import DQNAgent
import numpy as np

def test_integration():
    """환경과 에이전트 통합 테스트"""
    print("환경 + 에이전트 통합 테스트")
    
    # 환경 생성
    env = Game2048Env()
    
    # 에이전트 생성
    agent = DQNAgent(
        buffer_size=1000,
        batch_size=32,
        epsilon_decay=1000
    )
    
    print(f"환경 관찰 공간: {env.observation_space}")
    print(f"환경 액션 공간: {env.action_space}")
    
    # 몇 개의 에피소드 실행
    for episode in range(3):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 50:  # 최대 50 스텝
            # 에이전트 액션 선택
            action = agent.select_action(state)
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 경험 저장
            agent.store_experience(state, action, reward, next_state, done)
            
            # 학습 (버퍼가 충분히 찼을 때)
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss and steps % 10 == 0:
                    print(f"  Episode {episode+1}, Step {steps}: Loss = {loss:.6f}")
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"Episode {episode+1}: 총 보상 = {total_reward:.2f}, 스텝 = {steps}, 최종 점수 = {info['score']}")
    
    print("\n 통합 테스트 완료!")

if __name__ == "__main__":
    test_integration()
