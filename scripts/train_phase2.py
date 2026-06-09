"""
Phase 2 실행 스크립트.
  python scripts/train_phase2.py

순서:
  1. REINFORCE  (30k episodes)
  2. Actor-Critic (30k episodes)
  3. Phase 1 Q-Learning win_rates 로드
  4. 세 곡선 비교 그래프 저장
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.gomoku import GomokuEnv
from agents.policy_gradient import PolicyGradientAgent, self_play_train
from viz.plot_winrate import plot_comparison

# ------------------------------------------------------------------
# 하이퍼파라미터
# ------------------------------------------------------------------
BOARD_SIZE    = 6
N_IN_ROW      = 4
N_EPISODES    = 30_000
EVAL_INTERVAL = 300
EVAL_GAMES    = 300
LR_POLICY     = 1e-3
LR_VALUE      = 5e-3
GAMMA         = 0.95

ROOT      = os.path.join(os.path.dirname(__file__), "..")
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
PLOT_PATH = os.path.join(ROOT, "comparison_ph1_ph2.png")
Q_WR_PATH = os.path.join(CKPT_DIR, "q_phase1_winrates.json")
RF_PATH   = os.path.join(CKPT_DIR, "reinforce_phase2.pkl")
AC_PATH   = os.path.join(CKPT_DIR, "actorcritic_phase2.pkl")
RF_WR     = os.path.join(CKPT_DIR, "reinforce_phase2_winrates.json")
AC_WR     = os.path.join(CKPT_DIR, "actorcritic_phase2_winrates.json")

# ------------------------------------------------------------------

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    env = GomokuEnv(board_size=BOARD_SIZE, n_in_row=N_IN_ROW)

    print("=" * 55)
    print("Phase 2 — Policy Gradient (REINFORCE + Actor-Critic)")
    print(f"  보드: {BOARD_SIZE}×{BOARD_SIZE}  승리: {N_IN_ROW}목")
    print(f"  에피소드: {N_EPISODES:,}  α_π={LR_POLICY}  α_v={LR_VALUE}  γ={GAMMA}")
    print(f"  평가: 매 {EVAL_INTERVAL}판마다 랜덤봇 {EVAL_GAMES}게임")
    print("=" * 55)

    # ── 1. REINFORCE ────────────────────────────────────────────
    print("\n[ 1 / 2 ]  REINFORCE")
    rf_agent = PolicyGradientAgent(
        board_size=BOARD_SIZE, lr_policy=LR_POLICY,
        gamma=GAMMA, use_baseline=False,
    )
    rf_rates = self_play_train(env, rf_agent, N_EPISODES, EVAL_INTERVAL, EVAL_GAMES)
    rf_agent.save(RF_PATH)
    with open(RF_WR, "w") as f:
        json.dump(rf_rates, f)
    print(f"  → 저장: {RF_PATH}")
    print(f"  → 최종 승률: {rf_rates[-1]:.3f}")

    # ── 2. Actor-Critic ─────────────────────────────────────────
    print("\n[ 2 / 2 ]  Actor-Critic")
    ac_agent = PolicyGradientAgent(
        board_size=BOARD_SIZE, lr_policy=LR_POLICY, lr_value=LR_VALUE,
        gamma=GAMMA, use_baseline=True,
    )
    ac_rates = self_play_train(env, ac_agent, N_EPISODES, EVAL_INTERVAL, EVAL_GAMES)
    ac_agent.save(AC_PATH)
    with open(AC_WR, "w") as f:
        json.dump(ac_rates, f)
    print(f"  → 저장: {AC_PATH}")
    print(f"  → 최종 승률: {ac_rates[-1]:.3f}")

    # ── 3. Phase 1 Q-Learning 승률 로드 ─────────────────────────
    if os.path.exists(Q_WR_PATH):
        with open(Q_WR_PATH) as f:
            q_rates = json.load(f)
        print(f"\nQ-Learning 승률 로드: {len(q_rates)}개 포인트")
    else:
        q_rates = []
        print(f"\n[경고] {Q_WR_PATH} 없음 — Q-Learning 곡선 제외")

    # ── 4. 비교 그래프 ────────────────────────────────────────────
    series = [
        (q_rates,  EVAL_INTERVAL, "Q-Learning  (Phase 1)", "steelblue"),
        (rf_rates, EVAL_INTERVAL, "REINFORCE   (Phase 2)", "darkorange"),
        (ac_rates, EVAL_INTERVAL, "Actor-Critic (Phase 2)", "forestgreen"),
    ]
    plot_comparison(series, save_path=PLOT_PATH)

    print("\n=== Phase 2 완료 ===")
    print(f"  Q-Learning  최종 승률: {q_rates[-1]:.3f}" if q_rates else "  Q-Learning: 없음")
    print(f"  REINFORCE   최종 승률: {rf_rates[-1]:.3f}")
    print(f"  Actor-Critic 최종 승률: {ac_rates[-1]:.3f}")


if __name__ == "__main__":
    main()
