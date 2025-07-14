import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_2048 import Game2048Env, IllegalMove
import numpy as np
import time

def interactive_test():
    """키보드로 직접 2048 게임 플레이"""
    print("=== 2048 인터랙티브 테스트 ===")
    print("조작법:")
    print("w: 위로 이동")
    print("d: 오른쪽으로 이동") 
    print("s: 아래로 이동")
    print("a: 왼쪽으로 이동")
    print("q: 게임 종료")
    print("=" * 30)
    
    env = Game2048Env()
    obs = env.reset()
    env.render()
    
    total_reward = 0
    steps = 0
    
    while True:
        try:
            key = input("다음 움직임을 입력하세요 (w/a/s/d/q): ").lower().strip()
        except KeyboardInterrupt:
            print("\n게임이 중단되었습니다.")
            break
        
        if key == 'q':
            print("게임을 종료합니다.")
            break
        elif key == 'w':
            action = 0
        elif key == 'd':
            action = 1
        elif key == 's':
            action = 2
        elif key == 'a':
            action = 3
        else:
            print("잘못된 입력입니다. w/a/s/d/q 중 하나를 입력하세요.")
            continue
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"\n--- Step {steps} ---")
        print(f"Action: {['위', '오른쪽', '아래', '왼쪽'][action]}")
        print(f"Reward: {reward}")
        print(f"Illegal Move: {info.get('illegal_move', False)}")
        env.render()
        
        if done:
            if info.get('illegal_move', False):
                print("❌ 불법 이동으로 게임 종료!")
            else:
                print("게임 오버!")
            print(f"최종 점수: {info['score']}")
            print(f"최대 타일: {info['highest']}")
            print(f"총 보상: {total_reward}")
            print(f"총 스텝: {steps}")
            break

def test_observation_types():
    """관찰 타입 테스트"""
    print("\n=== 관찰 타입 테스트 ===")
    
    # Layered 관찰 테스트
    print("\n1. Layered 관찰:")
    env_layered = Game2048Env()
    obs_layered = env_layered.reset()
    print(f"  - Shape: {obs_layered.shape}")
    print(f"  - Type: {obs_layered.dtype}")
    print(f"  - Active layers: {np.sum(obs_layered, axis=(0,1))}")
    
    board = env_layered.get_board()
    print(f"  - 초기 보드:\n{board}")
    
    # 활성화된 레이어 확인
    print("  - 활성화된 레이어:")
    for layer in range(16):
        active_positions = np.where(obs_layered[:, :, layer] == 1)
        if len(active_positions[0]) > 0:
            value = 2 ** (layer + 1)
            positions = list(zip(active_positions[0], active_positions[1]))
            print(f"    Layer {layer} (값 {value}): {positions}")

def test_environment_functionality():
    """환경 기본 기능 테스트"""
    print("\n=== 환경 기본 기능 테스트 ===")
    
    env = Game2048Env(max_steps=100)
    
    print("1. 초기화 테스트:")
    obs = env.reset()
    board = env.get_board()
    print(f"  - 초기 점수: {env.score}")
    print(f"  - 초기 스텝: {env.steps}")
    print(f"  - 초기 불법 이동 횟수: {env.num_illegal}")
    print(f"  - 초기 타일 수: {np.count_nonzero(board)}")
    print(f"  - 관찰 크기: {obs.shape}")
    print(f"  - 관찰 타입: {obs.dtype}")
    print(f"  - 최대 스텝: {env.max_steps}")
    print(f"  - 최대 불법 이동: {env.max_illegal}")
    print(f"  - 최대 타일: {env.max_tile}")
    print(f"  - 불법 이동 페널티: {env.illegal_move_reward}")
    
    print("\n2. 액션 공간 테스트:")
    print(f"  - 액션 공간: {env.action_space}")
    print(f"  - 가능한 액션: {list(range(env.action_space.n))}")
    
    print("\n3. Step 함수 테스트:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  - Step {i+1}: Action={action}, Reward={reward:.1f}, Done={done}")
        print(f"    Info: {info}")
        if done:
            print("    게임이 조기 종료되었습니다.")
            break

def test_illegal_move_handling():
    """불법 이동 처리 테스트"""
    print("\n=== 불법 이동 처리 테스트 ===")
    
    env = Game2048Env()
    
    # 불법 이동 페널티 설정 테스트
    env.set_illegal_move_reward(-1.0)
    print(f"불법 이동 페널티 설정: {env.illegal_move_reward}")
    
    # 움직일 수 없는 보드 생성
    impossible_board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2]
    ])
    
    env.reset()
    env.set_board(impossible_board)
    print("불법 이동 테스트용 보드:")
    env.render()
    
    # 연속 불법 이동 테스트
    print("\n연속 불법 이동 테스트:")
    for i in range(15):  # 최대 불법 이동 횟수 이상 시도
        action = i % 4  # 0, 1, 2, 3 순환
        action_name = ['위', '오른쪽', '아래', '왼쪽'][action]
        
        obs, reward, done, info = env.step(action)
        print(f"시도 {i+1} - {action_name}: Reward={reward}, Done={done}, "
              f"Illegal={info.get('illegal_move', False)}, "
              f"불법횟수={env.num_illegal}, Steps={info.get('steps', 0)}")
        
        if done:
            if env.num_illegal >= env.max_illegal:
                print(f"✅ 최대 불법 이동 횟수({env.max_illegal})에 도달하여 게임 종료")
            elif info.get('steps', 0) >= env.max_steps:
                print(f"✅ 최대 스텝({env.max_steps})에 도달하여 게임 종료")
            else:
                print("❓ 예상치 못한 게임 종료")
            break
    else:
        print("❌ 게임이 종료되지 않았습니다")

def test_reward_system():
    """보상 시스템 테스트"""
    print("\n=== 보상 시스템 테스트 ===")
    
    env = Game2048Env()
    
    # 1. 정상 병합 보상 테스트
    print("1. 정상 병합 보상 테스트:")
    test_board = np.array([
        [2, 2, 0, 0],
        [4, 4, 0, 0],
        [8, 8, 0, 0],
        [0, 0, 0, 0]
    ])
    
    env.reset()
    env.set_board(test_board)
    prev_score = env.score
    
    print("테스트 보드:")
    env.render()
    
    # 왼쪽으로 이동 (병합 발생)
    obs, reward, done, info = env.step(3)  # 왼쪽
    
    print(f"왼쪽 이동 후:")
    print(f"  - 이전 점수: {prev_score}")
    print(f"  - 현재 점수: {env.score}")
    print(f"  - 보상: {reward}")
    print(f"  - 점수 증가: {env.score - prev_score}")
    print(f"  - 예상 점수: {4 + 8 + 16} (2+2=4, 4+4=8, 8+8=16)")
    env.render()
    
    # 2. 불법 이동 보상 테스트
    print("\n2. 불법 이동 보상 테스트:")
    env.set_illegal_move_reward(-5.0)
    
    impossible_board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2]
    ])
    
    env.reset()
    env.set_board(impossible_board)
    
    obs, reward, done, info = env.step(0)  # 불법 이동
    print(f"  - 불법 이동 보상: {reward}")
    print(f"  - 설정된 페널티: {env.illegal_move_reward}")
    print(f"  - 일치 여부: {'✅' if reward == env.illegal_move_reward else '❌'}")

def test_game_over_detection():
    """게임 종료 감지 테스트"""
    print("\n=== 게임 종료 감지 테스트 ===")
    
    env = Game2048Env()
    
    # 1. 빈 공간이 있는 경우 (게임 계속)
    board_with_space = np.array([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 64, 0]  # 빈 공간 있음
    ])
    
    env.reset()
    env.set_board(board_with_space)
    is_over = env._is_game_over()
    print(f"1. 빈 공간 있는 보드: 게임 종료 = {is_over}")
    
    # 2. 병합 가능한 경우 (게임 계속)
    board_mergeable = np.array([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 64, 64]  # 인접한 같은 숫자
    ])
    
    env.set_board(board_mergeable)
    is_over = env._is_game_over()
    print(f"2. 병합 가능한 보드: 게임 종료 = {is_over}")
    
    # 3. 게임 종료 상황
    board_game_over = np.array([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [16, 32, 64, 128]
    ])
    
    env.set_board(board_game_over)
    is_over = env._is_game_over()
    print(f"3. 게임 종료 보드: 게임 종료 = {is_over}")

def test_layered_observation_correctness():
    """레이어드 관찰 정확성 테스트"""
    print("\n=== 레이어드 관찰 정확성 테스트 ===")
    
    env = Game2048Env()
    
    # 다양한 값을 포함한 테스트 보드
    test_board = np.array([
        [2, 4, 8, 16],
        [32, 64, 128, 256],
        [512, 1024, 2048, 0],
        [0, 0, 0, 0]
    ])
    
    env.reset()
    env.set_board(test_board)
    obs = env._get_observation()
    
    print(f"테스트 보드:\n{test_board}")
    print(f"관찰 shape: {obs.shape}")
    print(f"관찰 dtype: {obs.dtype}")
    
    # 각 위치별로 레이어 확인
    all_correct = True
    for i in range(4):
        for j in range(4):
            value = test_board[i][j]
            if value > 0:
                expected_layer = int(np.log2(value)) - 1
                active_layers = np.where(obs[i][j] == 1)[0]
                
                if len(active_layers) == 1 and active_layers[0] == expected_layer:
                    status = "✅"
                else:
                    status = "❌"
                    all_correct = False
                
                print(f"위치 ({i},{j}), 값 {value}: 레이어 {expected_layer} {status}")
    
    print(f"\n전체 정확성: {'✅ 모두 정확' if all_correct else '❌ 오류 발견'}")

def benchmark_environment():
    """환경 성능 벤치마크"""
    print("\n=== 환경 성능 벤치마크 ===")
    
    episodes = 100
    max_steps = 50
    
    # Layered 관찰 벤치마크
    print("\n1. Layered 관찰 벤치마크:")
    env_layered = Game2048Env()
    start_time = time.time()
    total_steps = 0
    
    for episode in range(episodes):
        env_layered.reset()
        for step in range(max_steps):
            action = env_layered.action_space.sample()
            obs, reward, done, info = env_layered.step(action)
            total_steps += 1
            if done:
                break
    
    layered_time = time.time() - start_time
    print(f"  - {episodes} 에피소드, {total_steps} 스텝")
    print(f"  - 소요 시간: {layered_time:.2f}초")
    print(f"  - 초당 스텝: {total_steps / layered_time:.1f} steps/sec")

def test_error_handling():
    """에러 처리 테스트"""
    print("\n=== 에러 처리 테스트 ===")
    
    # 1. 잘못된 생성자 매개변수
    print("1. 잘못된 생성자 매개변수:")
    try:
        env = Game2048Env(size=0)
        print("  - size=0: ❌ 에러가 발생하지 않음")
    except ValueError as e:
        print(f"  - size=0: ✅ {e}")
    
    try:
        env = Game2048Env(max_steps=0)
        print("  - max_steps=0: ❌ 에러가 발생하지 않음")
    except ValueError as e:
        print(f"  - max_steps=0: ✅ {e}")
    
    # 2. 초기화 전 사용
    print("\n2. 초기화 전 사용:")
    env = Game2048Env()
    try:
        env.step(0)
        print("  - 초기화 전 step: ❌ 에러가 발생하지 않음")
    except RuntimeError as e:
        print(f"  - 초기화 전 step: ✅ {e}")
    
    # 3. 잘못된 액션
    print("\n3. 잘못된 액션:")
    env.reset()
    try:
        env.step(4)  # 유효 범위: 0-3
        print("  - 잘못된 액션: ❌ 에러가 발생하지 않음")
    except ValueError as e:
        print(f"  - 잘못된 액션: ✅ {e}")
    
    # 4. 잘못된 보드 크기
    print("\n4. 잘못된 보드 크기:")
    try:
        wrong_board = np.array([[1, 2], [3, 4]])  # 2x2 보드
        env.set_board(wrong_board)
        print("  - 잘못된 보드 크기: ❌ 에러가 발생하지 않음")
    except ValueError as e:
        print(f"  - 잘못된 보드 크기: ✅ {e}")
    
    # 5. 잘못된 최대 타일 설정
    print("\n5. 잘못된 최대 타일 설정:")
    try:
        env.set_max_tile(100)  # 2의 거듭제곱이 아님
        print("  - 2의 거듭제곱이 아닌 max_tile: ❌ 에러가 발생하지 않음")
    except ValueError as e:
        print(f"  - 2의 거듭제곱이 아닌 max_tile: ✅ {e}")
    
    try:
        env.set_max_tile(-64)  # 음수
        print("  - 음수 max_tile: ❌ 에러가 발생하지 않음")
    except ValueError as e:
        print(f"  - 음수 max_tile: ✅ {e}")

def test_step_limits():
    """스텝 제한 테스트"""
    print("\n=== 스텝 제한 테스트 ===")
    
    # 짧은 스텝 제한으로 테스트
    env = Game2048Env(max_steps=10)
    env.reset()
    
    print(f"최대 스텝 설정: {env.max_steps}")
    
    step_count = 0
    while step_count < 15:  # 최대 스텝보다 많이 시도
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        print(f"Step {step_count}: Action={action}, Done={done}, Steps={info.get('steps', 0)}")
        
        if done:
            if info.get('steps', 0) >= env.max_steps:
                print(f"✅ 최대 스텝({env.max_steps})에 도달하여 게임 종료")
            else:
                print(f"게임 종료 (다른 이유): {info}")
            break
    else:
        print("❌ 게임이 종료되지 않았습니다")

def test_max_tile_feature():
    """최대 타일 기능 테스트"""
    print("\n=== 최대 타일 기능 테스트 ===")
    
    env = Game2048Env()
    
    # 최대 타일 설정 테스트
    print("1. 최대 타일 설정 테스트:")
    env.set_max_tile(64)
    print(f"  - 최대 타일 설정: {env.max_tile}")
    
    # 64 타일이 있는 보드 생성
    test_board = np.array([
        [2, 4, 8, 16],
        [32, 64, 0, 0],  # 목표 타일 달성
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    env.reset()
    env.set_board(test_board)
    print("2. 최대 타일 달성 보드:")
    env.render()
    
    # 한 스텝 실행 후 게임 종료 확인
    obs, reward, done, info = env.step(1)  # 오른쪽 이동
    print(f"3. 결과: Done={done}, Highest={info.get('highest', 0)}")
    
    if done and info.get('highest', 0) >= env.max_tile:
        print("✅ 최대 타일 달성으로 게임 종료")
    else:
        print("❌ 최대 타일 기능이 작동하지 않음")

def test_rendering():
    """렌더링 기능 테스트"""
    print("\n=== 렌더링 기능 테스트 ===")
    
    env = Game2048Env()
    env.reset()
    
    # 몇 번의 이동 후 렌더링 테스트
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\n{i+1}번째 이동 후 렌더링:")
        rendered = env.render(mode='ansi')
        if rendered:
            print("✅ ANSI 모드 렌더링 성공")
            # 렌더링 결과에 새로운 정보들이 포함되었는지 확인
            expected_info = ['Score:', 'Steps:', 'Illegal:', 'Highest:', 'Empty:']
            for info_item in expected_info:
                if info_item in rendered:
                    print(f"  - {info_item} ✅")
                else:
                    print(f"  - {info_item} ❌")
        
        if done:
            break
    
    # Human 모드 테스트
    print("\nHuman 모드 렌더링 테스트:")
    try:
        env.render(mode='human')
        print("✅ Human 모드 렌더링 성공")
    except Exception as e:
        print(f"❌ Human 모드 렌더링 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("2048 Gym 환경 테스트 메뉴:")
    print("1. 인터랙티브 플레이")
    print("2. 관찰 타입 테스트")
    print("3. 환경 기본 기능 테스트")
    print("4. 불법 이동 처리 테스트")
    print("5. 보상 시스템 테스트")
    print("6. 게임 종료 감지 테스트")
    print("7. 레이어드 관찰 정확성 테스트")
    print("8. 성능 벤치마크")
    print("9. 에러 처리 테스트")
    print("10. 스텝 제한 테스트")
    print("11. 최대 타일 기능 테스트")
    print("12. 렌더링 기능 테스트")
    print("0. 모든 자동화 테스트 실행")
    
    choice = input("\n선택하세요 (0-12): ").strip()
    
    if choice == '1':
        interactive_test()
    elif choice == '2':
        test_observation_types()
    elif choice == '3':
        test_environment_functionality()
    elif choice == '4':
        test_illegal_move_handling()
    elif choice == '5':
        test_reward_system()
    elif choice == '6':
        test_game_over_detection()
    elif choice == '7':
        test_layered_observation_correctness()
    elif choice == '8':
        benchmark_environment()
    elif choice == '9':
        test_error_handling()
    elif choice == '10':
        test_step_limits()
    elif choice == '11':
        test_max_tile_feature()
    elif choice == '12':
        test_rendering()
    elif choice == '0':
        print(" 모든 자동화 테스트를 실행합니다...\n")
        test_observation_types()
        test_environment_functionality()
        test_illegal_move_handling()
        test_reward_system()
        test_game_over_detection()
        test_layered_observation_correctness()
        test_step_limits()
        test_max_tile_feature()
        test_rendering()  # 추가
        benchmark_environment()
        test_error_handling()
        print("\n 모든 테스트 완료!")
    else:
        print("잘못된 선택입니다. 환경 기본 기능 테스트를 실행합니다.")
        test_environment_functionality()

if __name__ == "__main__":
    main()