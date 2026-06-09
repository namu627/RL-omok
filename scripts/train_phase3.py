"""
Phase 3 학습 스크립트 — AlphaZero (6×6/4목).

  python -X utf8 scripts/train_phase3.py

결과:
  checkpoints/az_6x6_final.pt          최종 모델
  checkpoints/az_6x6_winrates.json     승률 기록
  az_phase3_winrate.png                 승률 곡선
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.alphazero.trainer import AlphaZeroTrainer
from viz.plot_winrate import plot_winrate

# ──────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 (6×6 빠른 검증용)
# ──────────────────────────────────────────────────────────────────────
BOARD_SIZE   = 6
N_IN_ROW     = 4
N_RES_BLOCKS = 3
N_FILTERS    = 64

N_SIM        = 100    # MCTS 시뮬레이션 수 (CPU 속도 고려해 줄임)
C_PUCT       = 1.5

N_ITERATIONS          = 300   # self-play → train 반복
SELF_PLAY_PER_ITER    = 20    # 매 반복 당 self-play 게임 수
BATCH_SIZE            = 512
TRAIN_STEPS_PER_ITER  = 5
EVAL_INTERVAL         = 10    # 10 반복마다 평가
EVAL_GAMES            = 100
TEMPERATURE_CUTOFF    = 10    # 10수 이후 greedy

LR      = 2e-3
L2_REG  = 1e-4

ROOT     = os.path.join(os.path.dirname(__file__), "..")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
PLOT_PATH = os.path.join(ROOT, "az_phase3_winrate.png")
WR_PATH   = os.path.join(CKPT_DIR, "az_6x6_winrates.json")

# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3 — AlphaZero (CNN + MCTS)")
    print(f"  보드: {BOARD_SIZE}×{BOARD_SIZE}  승리: {N_IN_ROW}목")
    print(f"  네트워크: ResBlocks={N_RES_BLOCKS}  Filters={N_FILTERS}")
    print(f"  MCTS 시뮬레이션: {N_SIM}  c_puct={C_PUCT}")
    print(f"  반복: {N_ITERATIONS}  self-play/iter={SELF_PLAY_PER_ITER}")
    print(f"  평가: {EVAL_INTERVAL}반복마다 랜덤봇 {EVAL_GAMES}게임")
    print("=" * 60)

    trainer = AlphaZeroTrainer(
        board_size=BOARD_SIZE,
        n_in_row=N_IN_ROW,
        n_res_blocks=N_RES_BLOCKS,
        n_filters=N_FILTERS,
        n_simulations=N_SIM,
        c_puct=C_PUCT,
        n_iterations=N_ITERATIONS,
        self_play_games_per_iter=SELF_PLAY_PER_ITER,
        batch_size=BATCH_SIZE,
        train_steps_per_iter=TRAIN_STEPS_PER_ITER,
        eval_interval=EVAL_INTERVAL,
        eval_games=EVAL_GAMES,
        lr=LR,
        l2_reg=L2_REG,
        ckpt_dir=CKPT_DIR,
        device="cpu",
        temperature_cutoff=TEMPERATURE_CUTOFF,
    )

    win_rates = trainer.train()

    # 승률 기록 저장
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(WR_PATH, "w", encoding="utf-8") as f:
        json.dump(win_rates, f)
    print(f"\n승률 기록 저장: {WR_PATH}")

    # 그래프
    if win_rates:
        plot_winrate(
            win_rates,
            eval_interval=EVAL_INTERVAL,
            title=f"AlphaZero Phase 3 ({BOARD_SIZE}×{BOARD_SIZE}/{N_IN_ROW}목) Win Rate vs Random",
            save_path=PLOT_PATH,
            smooth_window=5,
        )
        print(f"최종 승률: {win_rates[-1]:.3f}")

    print("\n=== Phase 3 완료 ===")


if __name__ == "__main__":
    main()
