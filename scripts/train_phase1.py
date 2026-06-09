"""
Phase 1 실행 스크립트.
  python scripts/train_phase1.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.gomoku import GomokuEnv
from agents.q_learning import QLearningAgent, self_play_train
from viz.plot_winrate import plot_winrate

# ------------------------------------------------------------------
# 하이퍼파라미터
# ------------------------------------------------------------------
BOARD_SIZE    = 6
N_IN_ROW      = 4
ALPHA         = 0.1       # 학습률
GAMMA         = 0.95      # 할인율
N_EPISODES    = 30_000    # 총 학습 에피소드
EPS_START     = 1.0       # 초기 탐험율
EPS_END       = 0.05      # 최소 탐험율
EPS_DECAY     = 0.8       # 이 비율까지 ε 선형 감소
EVAL_INTERVAL = 300       # 몇 판마다 평가전
EVAL_GAMES    = 300       # 평가전 판 수 (많을수록 노이즈 감소)
SAVE_PATH     = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "q_phase1.pkl")
PLOT_PATH     = os.path.join(os.path.dirname(__file__), "..", "winrate_phase1.png")

# ------------------------------------------------------------------

def main():
    env   = GomokuEnv(board_size=BOARD_SIZE, n_in_row=N_IN_ROW)
    agent = QLearningAgent(alpha=ALPHA, gamma=GAMMA)

    print("=" * 50)
    print("Phase 1 — Q-Learning Self-Play")
    print(f"  보드: {BOARD_SIZE}×{BOARD_SIZE}  승리조건: {N_IN_ROW}목")
    print(f"  에피소드: {N_EPISODES:,}  α={ALPHA}  γ={GAMMA}")
    print(f"  ε: {EPS_START} → {EPS_END} (처음 {int(EPS_DECAY*100)}% 동안 감소)")
    print(f"  평가: 매 {EVAL_INTERVAL}판마다 랜덤봇 {EVAL_GAMES}게임")
    print("=" * 50)

    win_rates = self_play_train(
        env=env,
        agent=agent,
        n_episodes=N_EPISODES,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay_ratio=EPS_DECAY,
        eval_interval=EVAL_INTERVAL,
        eval_games=EVAL_GAMES,
    )

    # 체크포인트 + win_rates 저장
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    agent.save(SAVE_PATH)
    import json
    wr_path = SAVE_PATH.replace(".pkl", "_winrates.json")
    with open(wr_path, "w") as f:
        json.dump(win_rates, f)
    print(f"\n모델 저장: {os.path.abspath(SAVE_PATH)}")

    # 승률 곡선 저장
    plot_winrate(win_rates, eval_interval=EVAL_INTERVAL, save_path=PLOT_PATH)

    print(f"\n=== 학습 완료 ===")
    print(f"  최종 승률 (vs 랜덤봇): {win_rates[-1]:.3f}")
    print(f"  Q-테이블 항목 수:      {len(agent.q_table):,}")


if __name__ == "__main__":
    main()
