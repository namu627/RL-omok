"""
최적화 전후 속도 비교 스모크 테스트.
3iter 학습 후 iter당 시간을 측정해 최적화 전 221s/iter와 비교.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.alphazero.trainer import AlphaZeroTrainer

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

trainer = AlphaZeroTrainer(
    board_size=15, n_in_row=5, renju=True,
    n_res_blocks=3, n_filters=64,
    n_simulations=50, c_puct=1.5,
    n_iterations=3,
    self_play_games_per_iter=5,
    batch_size=128, train_steps_per_iter=5,
    eval_interval=999,   # eval 없이 순수 self-play 속도만 측정
    eval_games=20,
    lr=2e-3, l2_reg=1e-4,
    ckpt_dir=CKPT_DIR, device="cpu",
    temperature_cutoff=20, ckpt_interval=0,
)

print("최적화 후 스모크 테스트 (3iter, n_sim=50, 5판/iter, renju=True)")
print("-" * 55)
t0 = time.time()
trainer.train()
total = time.time() - t0

avg = total / 3
print("-" * 55)
print(f"총 {total:.0f}s  |  iter 평균 {avg:.0f}s")
print(f"최적화 전 기준 221s/iter → 단축률 {(1 - avg/221)*100:.0f}%")
