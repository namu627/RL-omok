"""
scripts/profile_mcts.py
MCTS 병목 진단 — GPU predict 시간 vs MCTS Python 로직 시간 분리 측정.

Colab:  python scripts/profile_mcts.py
로컬:   python -X utf8 scripts/profile_mcts.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from agents.alphazero.network import AlphaZeroNet, board_to_tensor
from agents.alphazero.mcts import MCTS, MCTSNode
from env.gomoku import GomokuEnv, BLACK, WHITE

try:
    from env.renju_rules import classify_move as _renju_classify
except ImportError:
    from renju_rules import classify_move as _renju_classify

# ── 환경 ──────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BOARD_SIZE = 15
N_SIM      = 50
SP_GAMES   = 25

print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


def gpu_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def timed(fn, reps=1):
    """fn을 reps회 실행해 총 경과 시간(초) 반환. GPU sync 포함."""
    gpu_sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    gpu_sync()
    return time.perf_counter() - t


# ═══════════════════════════════════════════════════════════════════════════
# [1]  net.predict() 단일 호출 지연
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[1] net.predict() 단일 호출 지연  (batch=1)")
print("═" * 60)

board_sample = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
board_sample[7, 7] = 1

net_large = AlphaZeroNet(board_size=BOARD_SIZE, n_res_blocks=5, n_filters=128).to(DEVICE)
net_large.eval()

for label, nb, nf in [("3b×64ch  (로컬 검증)", 3, 64),
                       ("5b×128ch (Colab 설정)", 5, 128)]:
    _net = AlphaZeroNet(board_size=BOARD_SIZE, n_res_blocks=nb, n_filters=nf).to(DEVICE)
    _net.eval()
    for _ in range(20):
        _net.predict(board_sample, 1, device=DEVICE)
    gpu_sync()

    N = 300
    elapsed = timed(lambda: _net.predict(board_sample, 1, device=DEVICE), reps=N)
    ms = elapsed / N * 1000
    print(f"  {label}: {ms:.2f} ms/call")
    del _net


# ═══════════════════════════════════════════════════════════════════════════
# [2]  배치 크기별 처리량 + predict_batch vs N×predict 비교
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[2] 배치 크기별 처리량  (5b×128ch)")
print("═" * 60)

net_large.eval()
t1 = board_to_tensor(board_sample, 1).unsqueeze(0).to(DEVICE)

for bsz in [1, 4, 16, 25, 64, 256, 512]:
    batch = t1.expand(bsz, -1, -1, -1).contiguous()
    for _ in range(5):
        with torch.no_grad():
            net_large(batch)
    gpu_sync()

    reps = max(1, 200 // bsz)
    elapsed = timed(lambda: (None, net_large(batch))[0], reps=reps)  # noqa: B023
    per_sample_ms = elapsed / (reps * bsz) * 1000
    throughput    = reps * bsz / elapsed
    print(f"  batch={bsz:4d}: {per_sample_ms:.3f} ms/sample  |  {throughput:>8.0f} samples/s")

# predict_batch(25) vs 25×predict
print("\n  [predict_batch vs N×predict 비교]  (5b×128ch, 15×15, N=25)")
boards25  = [board_sample.copy() for _ in range(SP_GAMES)]
players25 = [BLACK] * SP_GAMES

# 워밍업
for _ in range(10):
    net_large.predict_batch(boards25, players25, device=DEVICE)
    net_large.predict(board_sample, BLACK, device=DEVICE)
gpu_sync()

N_BATCH = 200
t_batch = timed(lambda: net_large.predict_batch(boards25, players25, device=DEVICE),
                reps=N_BATCH)
t_seq   = timed(lambda: [net_large.predict(b, p, device=DEVICE)
                          for b, p in zip(boards25, players25)],
                reps=N_BATCH)
print(f"  predict_batch(25)  : {t_batch/N_BATCH*1000:.2f} ms/call")
print(f"  25 × predict(1)    : {t_seq/N_BATCH*1000:.2f} ms/call")
print(f"  배치 가속비        : {t_seq/t_batch:.1f}×")


# ═══════════════════════════════════════════════════════════════════════════
# [3]  Python 병목 컴포넌트별 단독 측정
#      classify_move / step_board / _expand / _select_leaf
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[3] Python 컴포넌트 단독 지연  (15×15 renju)")
print("═" * 60)

env = GomokuEnv(board_size=BOARD_SIZE, n_in_row=5, renju=True)
mcts_obj = MCTS(net_large, env, n_simulations=N_SIM, c_puct=1.5, device=DEVICE)

# 게임 중반 보드 (랜덤 15수 놓은 상태)
rng = np.random.default_rng(42)
mid_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
mid_cells = rng.choice(225, size=15, replace=False)
for k, idx in enumerate(mid_cells):
    mid_board[idx // BOARD_SIZE, idx % BOARD_SIZE] = BLACK if k % 2 == 0 else WHITE
mid_renju = env._to_renju_board(mid_board)
empty_cells = list(np.argwhere(mid_board == 0))

# (a) classify_move 단독
N_C = 500
t_classify = timed(
    lambda: [_renju_classify(mid_renju, int(r), int(c)) for r, c in empty_cells[:20]],
    reps=N_C
)
per_classify = t_classify / N_C / 20 * 1000
print(f"  (a) classify_move 1회   : {per_classify:.3f} ms")

# (b) step_board (BLACK)
empty_action = int(empty_cells[0][0]) * BOARD_SIZE + int(empty_cells[0][1])
N_SB = 2000
t_sb_black = timed(
    lambda: env.step_board(mid_board, empty_action, BLACK), reps=N_SB
)
t_sb_white = timed(
    lambda: env.step_board(mid_board, empty_action, WHITE), reps=N_SB
)
print(f"  (b) step_board(BLACK)   : {t_sb_black/N_SB*1000:.3f} ms  "
      f"(WHITE={t_sb_white/N_SB*1000:.3f} ms)")

# (c) _expand: 220개 action에 step_board 220회
env.current_player = BLACK
test_root = MCTSNode(
    parent=None, action=None,
    board=mid_board.copy(), current_player=BLACK,
)
raw_legal = mcts_obj._raw_legal_actions(mid_board)
policy_dummy = np.ones(225) / 225
N_EXP = 50
t_expand = timed(
    lambda: mcts_obj._expand(
        MCTSNode(parent=None, action=None, board=mid_board.copy(), current_player=BLACK),
        policy_dummy, raw_legal
    ),
    reps=N_EXP
)
print(f"  (c) _expand({len(raw_legal)}actions)  : {t_expand/N_EXP*1000:.1f} ms  "
      f"({t_expand/N_EXP/len(raw_legal)*1e6:.1f} μs/action × {len(raw_legal)})")

# (d) _select_leaf: 이미 확장된 트리에서 탐색
env.current_player = BLACK
root_expanded = MCTSNode(parent=None, action=None, board=mid_board.copy(), current_player=BLACK)
mcts_obj._expand(root_expanded, policy_dummy, raw_legal)
N_SL = 5000
t_sl = timed(lambda: mcts_obj._select_leaf(root_expanded), reps=N_SL)
print(f"  (d) _select_leaf        : {t_sl/N_SL*1000:.4f} ms  "
      f"({len(root_expanded.children)} children in root)")

# (e) _raw_legal_actions
N_RL = 5000
t_rl = timed(lambda: mcts_obj._raw_legal_actions(mid_board), reps=N_RL)
print(f"  (e) _raw_legal_actions  : {t_rl/N_RL*1000:.4f} ms")

# (f) legal_actions (renju BLACK — 금수 필터)
env.current_player = BLACK
N_LA = 500
t_la = timed(lambda: env.legal_actions(mid_board), reps=N_LA)
print(f"  (f) legal_actions(BLACK): {t_la/N_LA*1000:.2f} ms  "
      f"(≈{len(raw_legal)} positions × classify_move)")


# ═══════════════════════════════════════════════════════════════════════════
# [4]  직렬 MCTS (이전 방식) vs 병렬 while-loop 1회 직접 시뮬레이션
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[4] 직렬 MCTS 1회 (n_sim=50) vs 병렬 while-loop 1-step (sp=25)")
print("═" * 60)

start_board = env.reset()

# ── 직렬: get_action_probs 1회 ──────────────────────────────────────────
_pred_total = [0.0]; _pred_count = [0]
_orig_predict = net_large.predict
def _predict_timed(board, player, device="cpu"):
    gpu_sync(); t0 = time.perf_counter()
    r = _orig_predict(board, player, device=device)
    gpu_sync(); _pred_total[0] += time.perf_counter() - t0; _pred_count[0] += 1
    return r
net_large.predict = _predict_timed

mcts_obj2 = MCTS(net_large, env, n_simulations=N_SIM, c_puct=1.5, device=DEVICE)
# 워밍업
mcts_obj2.get_action_probs(start_board, BLACK, temperature=1.0)
_pred_total[0] = 0.0; _pred_count[0] = 0

REPS = 5; gpu_sync()
t0 = time.perf_counter()
for _ in range(REPS):
    mcts_obj2.get_action_probs(start_board, BLACK, temperature=1.0)
gpu_sync()
t_serial_total = (time.perf_counter() - t0) / REPS
t_serial_pred  = _pred_total[0] / REPS
n_serial_pred  = _pred_count[0] // REPS
t_serial_cpu   = t_serial_total - t_serial_pred

print(f"  직렬 MCTS 1회 (n_sim={N_SIM}):")
print(f"    총 시간         : {t_serial_total*1000:.1f} ms")
print(f"    net.predict()   : {t_serial_pred*1000:.1f} ms  "
      f"({n_serial_pred}회, {t_serial_pred/n_serial_pred*1000:.2f} ms/call)  "
      f"← {t_serial_pred/t_serial_total*100:.0f}%")
print(f"    Python CPU 로직 : {t_serial_cpu*1000:.1f} ms  ← {t_serial_cpu/t_serial_total*100:.0f}%")

net_large.predict = _orig_predict  # monkeypatch 해제

# ── 병렬: while-loop 1 iteration (sp=25 게임의 root build + n_sim rounds) ──
print(f"\n  병렬 while-loop 1 iteration (sp={SP_GAMES}, n_sim={N_SIM}):")

boards_p   = [start_board.copy() for _ in range(SP_GAMES)]
players_p  = [BLACK] * SP_GAMES

_pb_total = [0.0]; _pb_count = [0]; _pb_n_total = [0]
_orig_pb = net_large.predict_batch
def _pb_timed(boards, players, device="cpu"):
    gpu_sync(); t0 = time.perf_counter()
    r = _orig_pb(boards, players, device=device)
    gpu_sync(); _pb_total[0] += time.perf_counter() - t0
    _pb_count[0] += 1; _pb_n_total[0] += len(boards)
    return r
net_large.predict_batch = _pb_timed

# 워밍업
for _ in range(3):
    net_large.predict_batch(boards_p, players_p, device=DEVICE)
gpu_sync()
_pb_total[0] = 0.0; _pb_count[0] = 0; _pb_n_total[0] = 0

# 실제 측정: while-loop 1 step
_t_root = [0.0]; _t_sim = [0.0]; _t_action = [0.0]
active_idx = list(range(SP_GAMES))

gpu_sync()
t0_par = time.perf_counter()

# root batch build
t_rb = time.perf_counter()
root_policies, _ = net_large.predict_batch(boards_p, players_p, device=DEVICE)
roots_p = {}
for j, i in enumerate(active_idx):
    env.current_player = players_p[i]
    legal = env.legal_actions(boards_p[i])
    root = MCTSNode(parent=None, action=None, board=boards_p[i].copy(), current_player=players_p[i])
    pol = root_policies[j].copy()
    mask = np.zeros(225); mask[legal] = 1.0; pol *= mask
    s = pol.sum(); pol = pol / s if s > 0 else (np.zeros(225).__setitem__(legal, 1/len(legal)) or pol)
    mcts_obj._expand(root, pol, legal)
    roots_p[i] = root
_t_root[0] = time.perf_counter() - t_rb

# n_sim simulation rounds
batch_sizes = []
t_sim_start = time.perf_counter()
for _ in range(N_SIM):
    to_eval = []
    for i in active_idx:
        leaf, path = mcts_obj._select_leaf(roots_p[i])
        if leaf.is_terminal:
            mcts_obj._backup(path, mcts_obj._terminal_value(leaf))
        else:
            to_eval.append((i, leaf, path))
    batch_sizes.append(len(to_eval))
    if to_eval:
        ev_b = [l.board for _, l, _ in to_eval]
        ev_p = [l.current_player for _, l, _ in to_eval]
        policies, values = net_large.predict_batch(ev_b, ev_p, device=DEVICE)
        for (i, leaf, path), pol, val in zip(to_eval, policies, values):
            mcts_obj._expand_and_backup(leaf, path, pol, val)
_t_sim[0] = time.perf_counter() - t_sim_start

gpu_sync()
t_par_total = time.perf_counter() - t0_par
net_large.predict_batch = _orig_pb

print(f"    총 시간              : {t_par_total*1000:.1f} ms")
print(f"    root build           : {_t_root[0]*1000:.1f} ms  "
      f"({SP_GAMES}게임 root 빌드, legal_actions + _expand 포함)")
print(f"    n_sim={N_SIM} rounds  : {_t_sim[0]*1000:.1f} ms  "
      f"(predict_batch {_pb_count[0]}회 포함)")
print(f"    ├─ predict_batch()   : {_pb_total[0]*1000:.1f} ms  "
      f"({_pb_count[0]}회, avg batch={_pb_n_total[0]/max(_pb_count[0],1):.1f})")
print(f"    └─ Python 트리 로직  : {(_t_sim[0]-_pb_total[0])*1000:.1f} ms")
print(f"    실제 배치 크기 분포  : min={min(batch_sizes)}  "
      f"avg={sum(batch_sizes)/len(batch_sizes):.1f}  max={max(batch_sizes)}")


# ═══════════════════════════════════════════════════════════════════════════
# [5]  iter 시간 외삽
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[5] 실측 기반 iter 시간 예측")
print("═" * 60)

avg_moves_per_game = 50
t_while_iter_ms = t_par_total * 1000  # 병렬 while-loop 1회 소요 ms

for sp in [5, 25]:
    # 직렬 예측
    t_serial_iter = t_serial_total * sp * avg_moves_per_game
    # 병렬 예측: while-loop avg_moves_per_game 회 × (sp/SP_GAMES) 스케일
    t_parallel_iter = t_while_iter_ms / 1000 * avg_moves_per_game * (sp / SP_GAMES)
    print(f"  sp={sp:2d}: 직렬={t_serial_iter:.0f}s/iter  "
          f"병렬(while×{avg_moves_per_game})≈{t_parallel_iter:.0f}s/iter")

print(f"\n  (참고) 병렬 while-loop 1회 구성:")
root_frac = _t_root[0] / t_par_total * 100
sim_frac  = _t_sim[0]  / t_par_total * 100
pred_frac = _pb_total[0] / t_par_total * 100
cpu_frac  = (_t_sim[0] - _pb_total[0]) / t_par_total * 100
print(f"    root build      : {root_frac:.0f}%  (legal_actions × {SP_GAMES} + _expand × {SP_GAMES})")
print(f"    sim rounds      : {sim_frac:.0f}%")
print(f"      predict_batch : {pred_frac:.0f}%  ← GPU")
print(f"      Python 트리   : {cpu_frac:.0f}%  ← CPU (_select_leaf + _expand_and_backup)")

print("\n[결론]")
if cpu_frac > 50:
    print(f"  → 병목: Python CPU 트리 로직 ({cpu_frac:.0f}%)")
    print(f"       주범: _expand가 매 leaf마다 {len(raw_legal)}개 step_board 호출")
    print(f"            + BLACK 차례면 step_board마다 classify_move 호출")
    print(f"  → 해결 방향: Lazy Expansion (자식을 전부 미리 만들지 않고 선택 시 1개씩 생성)")
elif pred_frac > 50:
    print(f"  → 병목: GPU predict_batch ({pred_frac:.0f}%)")
    print(f"  → 해결 방향: 더 큰 배치 (sp_games 증가) 또는 n_sim 감소")
else:
    print(f"  → 병목: root build ({root_frac:.0f}%)")
    print(f"       legal_actions(BLACK)가 {len(raw_legal)}회 classify_move 호출")
