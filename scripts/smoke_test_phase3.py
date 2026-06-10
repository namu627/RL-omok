"""
Phase 3 파이프라인 스모크 테스트.

목적: "에러 없이 끝까지 돈다" 확인 (실력 향상 불필요).
  - self-play 5게임 × 2회 반복
  - MCTS 시뮬레이션 10회
  - 평가 10판
  - 체크포인트 저장

실행:
  python -X utf8 scripts/smoke_test_phase3.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.alphazero.trainer import AlphaZeroTrainer

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

def main():
    print("=" * 55)
    print("Phase 3 스모크 테스트")
    print("  6×6/4목 | MCTS 10sim | 2 iter | self-play 5게임/iter")
    print("=" * 55)
    t0 = time.time()

    trainer = AlphaZeroTrainer(
        board_size=6,
        n_in_row=4,
        n_res_blocks=3,
        n_filters=64,
        n_simulations=10,       # 아주 짧게
        c_puct=1.5,
        n_iterations=2,         # 2 사이클만
        self_play_games_per_iter=5,   # 게임 5판
        batch_size=32,          # 데이터 적으니 작게
        train_steps_per_iter=2,
        eval_interval=1,        # 매 iter 평가
        eval_games=10,          # 평가 10판
        lr=1e-3,
        l2_reg=1e-4,
        ckpt_dir=CKPT_DIR,
        device="cpu",
        temperature_cutoff=6,
    )

    print("\n[1/3] Self-play 데이터 수집 + 학습 루프 시작...")
    win_rates = trainer.train()

    elapsed = time.time() - t0
    print(f"\n[2/3] 루프 완료. 소요 시간: {elapsed:.1f}초")

    # 체크포인트 확인
    ckpt_path = os.path.join(CKPT_DIR, "az_6x6_final.pt")
    exists = os.path.isfile(ckpt_path)
    print(f"[3/3] 체크포인트 존재 여부: {'OK' if exists else 'FAIL'} ({ckpt_path})")

    print("\n" + "=" * 55)
    if exists and len(win_rates) > 0:
        print("스모크 테스트 PASS")
        print(f"  기록된 승률: {[f'{r:.3f}' for r in win_rates]}")
    else:
        print("스모크 테스트 FAIL — 위 로그를 확인하세요")
    print("=" * 55)

if __name__ == "__main__":
    main()
