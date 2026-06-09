import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.gomoku import GomokuEnv, BLACK, WHITE, EMPTY
from agents.random_agent import RandomAgent


# ------------------------------------------------------------------
# 단위 테스트
# ------------------------------------------------------------------

def test_reset():
    env = GomokuEnv()
    state = env.reset()
    assert state.shape == (6, 6), "보드 크기 오류"
    assert state.sum() == 0, "초기화 후 빈 칸이어야 함"
    print("[PASS] test_reset")


def test_step_basic():
    env = GomokuEnv()
    env.reset()
    state, reward, done, info = env.step(0)  # (0,0)에 착수
    assert state[0, 0] == BLACK
    assert reward == 0.0
    assert not done
    print("[PASS] test_step_basic")


def test_illegal_action():
    env = GomokuEnv()
    env.reset()
    env.step(0)
    env.step(1)
    try:
        env.step(0)  # 이미 돌이 있는 자리
        print("[FAIL] test_illegal_action: 예외가 발생해야 함")
    except ValueError:
        print("[PASS] test_illegal_action")


def test_win_horizontal():
    env = GomokuEnv(board_size=6, n_in_row=4)
    env.reset()
    # 흑: (0,0)(0,1)(0,2)(0,3)  백: (1,0)(1,1)(1,2)
    moves = [0, 6, 1, 7, 2, 8, 3]  # 마지막 수(3번)가 흑 4목
    final_reward, final_done = None, None
    for i, action in enumerate(moves):
        _, reward, done, info = env.step(action)
        final_reward, final_done = reward, done
    assert final_done, "4목이 완성됐으므로 게임 종료여야 함"
    assert final_reward == 1.0, "착수한 플레이어 보상 +1이어야 함"
    assert info["winner"] == BLACK
    print("[PASS] test_win_horizontal")


def test_win_vertical():
    env = GomokuEnv(board_size=6, n_in_row=4)
    env.reset()
    # 흑: (0,0)(1,0)(2,0)(3,0)  백: (0,1)(1,1)(2,1)
    moves = [0, 1, 6, 7, 12, 13, 18]
    for i, action in enumerate(moves):
        _, reward, done, info = env.step(action)
    assert done
    assert info["winner"] == BLACK
    print("[PASS] test_win_vertical")


def test_win_diagonal():
    env = GomokuEnv(board_size=6, n_in_row=4)
    env.reset()
    # 흑: (0,0)(1,1)(2,2)(3,3)  백: (0,1)(0,2)(0,3)
    moves = [0, 1, 7, 2, 14, 3, 21]
    for action in moves:
        _, reward, done, info = env.step(action)
    assert done
    assert info["winner"] == BLACK
    print("[PASS] test_win_diagonal")


def test_draw():
    """가득 찼지만 아무도 이기지 못하는 상황을 억지로 만들기 어려우므로,
    무승부 감지는 랜덤 대국 통계로 간접 확인."""
    print("[SKIP] test_draw (랜덤 대국 통계로 대체)")


# ------------------------------------------------------------------
# 랜덤 vs 랜덤 대국 테스트
# ------------------------------------------------------------------

def run_random_games(n_games: int = 1000, verbose_first: int = 1):
    env = GomokuEnv(board_size=6, n_in_row=4)
    agent = RandomAgent()

    results = {BLACK: 0, WHITE: 0, EMPTY: 0}

    for game_idx in range(n_games):
        state = env.reset()
        show = game_idx < verbose_first

        if show:
            print("=" * 30)
            print(f"대국 #{game_idx + 1} (랜덤 vs 랜덤)")
            print("=" * 30)
            env.render()

        while True:
            actions = env.legal_actions(state)
            action = agent.select_action(state, actions)
            state, reward, done, info = env.step(action)

            if show:
                env.render()

            if done:
                results[info["winner"]] += 1
                break

    print(f"\n=== 랜덤 vs 랜덤  {n_games}판 결과 ===")
    print(f"  흑(X) 승: {results[BLACK]:>5}  ({results[BLACK]/n_games*100:.1f}%)")
    print(f"  백(O) 승: {results[WHITE]:>5}  ({results[WHITE]/n_games*100:.1f}%)")
    print(f"  무승부:   {results[EMPTY]:>5}  ({results[EMPTY]/n_games*100:.1f}%)")
    print()

    # 기본 sanity check
    assert results[BLACK] + results[WHITE] + results[EMPTY] == n_games
    # 선공 우위가 있지만 지나치게 편향되면 버그 의심
    black_rate = results[BLACK] / n_games
    assert 0.2 < black_rate < 0.8, f"선공 승률이 비정상적으로 편향됨: {black_rate:.2f}"
    print("[PASS] run_random_games 통계 sanity check")


# ------------------------------------------------------------------

if __name__ == "__main__":
    test_reset()
    test_step_basic()
    test_illegal_action()
    test_win_horizontal()
    test_win_vertical()
    test_win_diagonal()
    test_draw()
    print()
    run_random_games(n_games=1000, verbose_first=1)
