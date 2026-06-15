"""
tests/test_mcts_parallel.py
인터 게임 배치 병렬 self-play 정확성 테스트.

핵심 보장:
  (1) test_mcts_inter_game_independence  — 병렬 게임 A의 MCTS N분포 == 직렬 게임 A
      + lazy expansion 경로 실제 사용 검증 (root.children < n_legal)
  (2) test_lazy_equals_eager_expansion   — lazy N분포 == eager(_expand 선생성) N분포
      → "자식을 언제 만드느냐"만 다르고 PUCT 결정은 동일함을 직접 증명
  (3) test_inter_game_no_cross_contamination — B 시뮬레이션이 A 트리를 변경하지 않음
  (4) test_parallel_root_N_equals_n_sim  — 병렬 실행 후 각 루트 N = n_sim
  (5~8) TestParallelCollectIntegrity     — trainer._collect_self_play 무결성

실행: python -m pytest tests/test_mcts_parallel.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from agents.alphazero.network import AlphaZeroNet
from agents.alphazero.mcts import MCTS, MCTSNode
from agents.alphazero.trainer import AlphaZeroTrainer
from env.gomoku import GomokuEnv, BLACK, WHITE


# ──────────────────────────────────────────────────────────────────────
# [1] MCTS 트리 독립성 + Lazy Expansion 정확성
# ──────────────────────────────────────────────────────────────────────

class TestInterGameIndependence:

    def test_mcts_inter_game_independence(self):
        """게임 A를 게임 B와 병렬(lockstep)로 돌린 N분포가
           게임 A를 단독 직렬로 돌린 결과와 완전히 동일.

        Lazy expansion 경로 사용 검증도 포함:
          - _build_root 후 root.children == 0 (자식 선생성 없음)
          - N_SIM 시뮬 후 root.children < n_legal (lazy: 방문분만 생성)
        """
        torch.manual_seed(42)
        net  = AlphaZeroNet(board_size=6, n_res_blocks=2, n_filters=32)
        env  = GomokuEnv(board_size=6, n_in_row=4)
        N_SIM = 25
        mcts = MCTS(net, env, n_simulations=N_SIM, device="cpu")

        # 흑 돌 하나가 놓인 6×6 보드 (백 차례)
        board = np.zeros((6, 6), dtype=np.int8)
        board[2, 2] = BLACK

        # ── 직렬: 게임 A 단독 ──────────────────────────────────────
        env.current_player = WHITE
        root_serial = mcts._build_root(board.copy(), WHITE)

        # lazy expansion 검증 #1: _build_root 직후 자식이 없어야 함
        assert root_serial._policy is not None, "_build_root: _policy 설정 안 됨"
        assert len(root_serial.children) == 0, (
            f"_build_root 후 자식 {len(root_serial.children)}개 — eager expand 호출된 것으로 보임"
        )

        for _ in range(N_SIM):
            mcts._simulate(root_serial)
        N_serial = {a: c.N for a, c in root_serial.children.items()}

        # root.N = N_SIM 검증
        assert root_serial.N == N_SIM, (
            f"직렬 루트 방문 횟수 오류: {root_serial.N} ≠ {N_SIM}"
        )

        # lazy expansion 검증 #2: N_SIM 후 자식 수 < n_legal
        # (eager였다면 legal 전부 = 35개 선생성; lazy면 최대 N_SIM = 25개)
        n_legal = len(root_serial._legal_actions)  # WHITE: 금수 없음 → 35
        assert len(root_serial.children) <= N_SIM < n_legal, (
            f"lazy 경로 검증 실패: root.children={len(root_serial.children)}, "
            f"N_SIM={N_SIM}, n_legal={n_legal}"
        )

        # ── 병렬: 게임 A와 게임 B 동시 (lockstep) ─────────────────
        env.current_player = WHITE
        root_A = mcts._build_root(board.copy(), WHITE)
        env.current_player = WHITE
        root_B = mcts._build_root(board.copy(), WHITE)   # 동일 초기 상태

        assert len(root_A.children) == 0 and len(root_B.children) == 0, \
            "병렬 루트도 _build_root 직후 자식이 없어야 함"

        for _ in range(N_SIM):
            to_eval = []
            for root in [root_A, root_B]:
                leaf, path = mcts._select_leaf(root)
                if leaf.is_terminal:
                    mcts._backup(path, mcts._terminal_value(leaf))
                else:
                    to_eval.append((leaf, path))

            if to_eval:
                ev_boards  = [l.board for l, _ in to_eval]
                ev_players = [l.current_player for l, _ in to_eval]
                policies, values = net.predict_batch(
                    ev_boards, ev_players, device="cpu"
                )
                for (leaf, path), pol, val in zip(to_eval, policies, values):
                    mcts._expand_and_backup(leaf, path, pol, val)

        N_A = {a: c.N for a, c in root_A.children.items()}
        N_B = {a: c.N for a, c in root_B.children.items()}

        # 병렬 루트도 lazy 경로 확인
        assert len(root_A.children) <= N_SIM < n_legal and \
               len(root_B.children) <= N_SIM < n_legal, \
            "병렬 루트가 lazy 경로를 사용하지 않음"

        # 독립성 검증: 병렬 게임 A == 직렬
        assert N_A == N_serial, (
            "병렬 게임 A의 N분포가 직렬과 다름\n"
            f"serial  상위 5: {sorted(N_serial.items(), key=lambda x: -x[1])[:5]}\n"
            f"parallel상위 5: {sorted(N_A.items(),     key=lambda x: -x[1])[:5]}"
        )
        # 같은 초기 상태 → B도 동일
        assert N_B == N_serial, (
            "같은 초기 상태로 시작한 병렬 게임 B의 N분포가 직렬과 다름"
        )

    def test_lazy_equals_eager_expansion(self):
        """Lazy expansion과 Eager expansion(_expand 선생성)이 동일한 N 분포를 생성한다.

        이것이 lazy expansion 정확성의 핵심 증거:
          - Lazy:  PUCT가 선택할 때만 _create_child (step_board 1회)
          - Eager: legal 자식 전부 _expand로 선생성 후 동일 PUCT 적용
          - PUCT 점수 공식이 수학적으로 동일:
              방문 자식: -Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
              미방문(lazy)/선생성N=0(eager): c_puct * P * sqrt(N_parent)  ← 동일!
          따라서 같은 시드·보드에서 두 방식의 N 분포는 정수 단위로 같아야 한다.
        """
        torch.manual_seed(42)
        net  = AlphaZeroNet(board_size=6, n_res_blocks=2, n_filters=32)
        env  = GomokuEnv(board_size=6, n_in_row=4)
        N_SIM = 25
        mcts = MCTS(net, env, n_simulations=N_SIM, device="cpu")

        board = np.zeros((6, 6), dtype=np.int8)
        board[2, 2] = BLACK
        n_legal = 35  # WHITE 차례: 6×6 - 1 = 35개 빈 칸 = legal 수

        # ── Lazy 방식 (현재 기본) ─────────────────────────────────
        env.current_player = WHITE
        root_lazy = mcts._build_root(board.copy(), WHITE)

        # lazy 검증: 자식 선생성 없음
        assert root_lazy._policy is not None
        assert len(root_lazy.children) == 0, "lazy root: 자식이 미리 생성되면 안 됨"
        assert len(root_lazy._legal_actions) == n_legal

        for _ in range(N_SIM):
            mcts._simulate(root_lazy)

        assert root_lazy.N == N_SIM
        # lazy: 방문한 자식만 children에 있음 (최대 N_SIM개, n_legal보다 적음)
        assert len(root_lazy.children) <= N_SIM < n_legal, (
            f"lazy 루트 자식 수 오류: {len(root_lazy.children)}"
        )
        N_lazy = {a: c.N for a, c in root_lazy.children.items()}

        # ── Eager 방식 (기존 방식 재현: _expand로 자식 선생성) ────
        env.current_player = WHITE
        root_eager = mcts._build_root(board.copy(), WHITE)
        # _build_root 후 _expand로 모든 legal 자식을 미리 생성
        mcts._expand(root_eager, root_eager._policy.copy(), root_eager._legal_actions)

        # eager 검증: 모든 legal 자식 선생성됨
        assert len(root_eager.children) == n_legal, (
            f"eager root: 자식 수 {len(root_eager.children)} ≠ n_legal {n_legal}"
        )
        # 선생성된 자식은 아직 평가 안 됨 (N=0, _policy=None)
        assert all(c.N == 0 for c in root_eager.children.values()), \
            "eager 선생성 자식은 초기 N=0이어야 함"
        assert all(c._policy is None for c in root_eager.children.values()), \
            "eager 선생성 자식은 아직 평가 전(_policy=None)이어야 함"

        for _ in range(N_SIM):
            mcts._simulate(root_eager)

        assert root_eager.N == N_SIM
        N_eager_all      = {a: c.N for a, c in root_eager.children.items()}
        N_eager_visited  = {a: n for a, n in N_eager_all.items() if n > 0}

        # N 분포 정수 단위 비교
        # lazy:  방문한 자식만 dict에 존재  → {action: N}
        # eager: 전체 자식 중 N>0인 것만   → {action: N}
        # 같은 PUCT → 같은 방문 순서 → 같은 N
        assert N_lazy == N_eager_visited, (
            "Lazy와 Eager expansion의 N분포가 다름 — PUCT 동등성 깨짐\n"
            f"lazy   상위 5: {sorted(N_lazy.items(),         key=lambda x: -x[1])[:5]}\n"
            f"eager  상위 5: {sorted(N_eager_visited.items(), key=lambda x: -x[1])[:5]}\n"
            f"lazy   총 방문: {sum(N_lazy.values())}\n"
            f"eager  총 방문: {sum(N_eager_visited.values())}"
        )

        # 추가: 방문 자식 수도 동일
        assert len(N_lazy) == len(N_eager_visited), (
            f"방문 자식 수 불일치: lazy={len(N_lazy)}, eager={len(N_eager_visited)}"
        )

    def test_inter_game_no_cross_contamination(self):
        """게임 B의 시뮬레이션이 게임 A의 트리를 전혀 변경하지 않는다."""
        torch.manual_seed(7)
        net  = AlphaZeroNet(board_size=6, n_res_blocks=2, n_filters=32)
        env  = GomokuEnv(board_size=6, n_in_row=4)
        mcts = MCTS(net, env, n_simulations=10, device="cpu")

        boardA = np.zeros((6, 6), dtype=np.int8)
        boardB = np.zeros((6, 6), dtype=np.int8)
        boardA[0, 0] = BLACK   # 서로 다른 초기 상태
        boardB[5, 5] = BLACK

        env.current_player = WHITE
        rootA = mcts._build_root(boardA.copy(), WHITE)
        env.current_player = WHITE
        rootB = mcts._build_root(boardB.copy(), WHITE)

        # rootA 자식들의 초기 스냅샷 (lazy: 아직 자식 없음)
        assert len(rootA.children) == 0, "lazy build 직후 rootA 자식 없어야 함"
        N_A_before = {a: c.N for a, c in rootA.children.items()}  # 빈 dict

        # 게임 B만 10번 시뮬레이션 (게임 A는 건드리지 않음)
        for _ in range(10):
            leaf, path = mcts._select_leaf(rootB)
            if leaf.is_terminal:
                mcts._backup(path, mcts._terminal_value(leaf))
            else:
                pol, val = net.predict(leaf.board, leaf.current_player)
                mcts._expand_and_backup(leaf, path, pol, val)

        N_A_after = {a: c.N for a, c in rootA.children.items()}

        assert N_A_after == N_A_before, (
            "게임 B 시뮬레이션이 게임 A 트리 노드 N 값을 변경함 — 트리 공유 버그"
        )
        # rootA 자체도 N=0 유지
        assert rootA.N == 0, (
            f"게임 B 시뮬레이션 후 게임 A 루트 N={rootA.N} ≠ 0"
        )

    def test_parallel_root_N_equals_n_sim(self):
        """병렬 실행 후 각 루트의 총 방문 수가 n_sim과 일치."""
        torch.manual_seed(99)
        net   = AlphaZeroNet(board_size=6, n_res_blocks=2, n_filters=32)
        env   = GomokuEnv(board_size=6, n_in_row=4)
        N_SIM = 15
        mcts  = MCTS(net, env, n_simulations=N_SIM, device="cpu")

        board = np.zeros((6, 6), dtype=np.int8)
        env.current_player = BLACK
        roots = [mcts._build_root(board.copy(), BLACK) for _ in range(3)]

        # lazy 검증: 모든 루트 자식 없음
        for idx, root in enumerate(roots):
            assert len(root.children) == 0, f"루트 {idx}: lazy build 후 자식 없어야 함"

        for _ in range(N_SIM):
            to_eval = []
            for root in roots:
                leaf, path = mcts._select_leaf(root)
                if leaf.is_terminal:
                    mcts._backup(path, mcts._terminal_value(leaf))
                else:
                    to_eval.append((leaf, path))
            if to_eval:
                ev_b = [l.board for l, _ in to_eval]
                ev_p = [l.current_player for l, _ in to_eval]
                pols, vals = net.predict_batch(ev_b, ev_p, device="cpu")
                for (leaf, path), pol, val in zip(to_eval, pols, vals):
                    mcts._expand_and_backup(leaf, path, pol, val)

        for idx, root in enumerate(roots):
            assert root.N == N_SIM, (
                f"루트 {idx}: N={root.N} ≠ n_sim={N_SIM}"
            )


# ──────────────────────────────────────────────────────────────────────
# [2] 에피소드 무결성 (trainer._collect_self_play)
# ──────────────────────────────────────────────────────────────────────

class TestParallelCollectIntegrity:

    def _make_trainer(self, seed: int, sp: int = 4, n_sim: int = 8,
                      board: int = 6, n_row: int = 4) -> AlphaZeroTrainer:
        torch.manual_seed(seed)
        np.random.seed(seed)   # Dirichlet 노이즈 재현성
        return AlphaZeroTrainer(
            board_size=board, n_in_row=n_row,
            n_simulations=n_sim, n_iterations=1,
            self_play_games_per_iter=sp,
            batch_size=32, device="cpu",
        )

    def test_sample_shapes_and_z_values(self):
        """각 샘플의 텐서 크기, 정책 합 ≈ 1, z ∈ {-1, 0, 1}."""
        trainer = self._make_trainer(seed=1, sp=3)
        data = trainer._collect_self_play()
        assert len(data) > 0, "데이터가 없음"

        valid_z = {-1.0, 0.0, 1.0}
        for state_t, pi, z in data:
            assert state_t.shape == (3, 6, 6), f"상태 shape 오류: {state_t.shape}"
            assert pi.shape == (36,),            f"정책 shape 오류: {pi.shape}"
            assert abs(pi.sum() - 1.0) < 1e-5,  f"정책 합 오류: {pi.sum():.8f}"
            assert z in valid_z,                 f"z 값 오류: {z}"

    def test_minimum_data_volume(self):
        """sp_games × 최소 1수 × 8증강 이상의 데이터 생성."""
        SP = 4
        trainer = self._make_trainer(seed=2, sp=SP)
        data = trainer._collect_self_play()
        min_expected = SP * 1 * 8
        assert len(data) >= min_expected, (
            f"데이터 수 부족: {len(data)} < {min_expected}"
        )

    def test_winner_reward_consistency(self):
        """게임이 실제로 종료되고 승패 z ∈ {+1, -1}이 존재한다.
        (무승부만 나오면 z={0}만 → 버그 징후)
        """
        # n_in_row=3으로 빠른 종료 유도
        trainer = self._make_trainer(seed=3, sp=4, n_sim=6,
                                     board=6, n_row=3)
        data = trainer._collect_self_play()
        z_values = {z for _, _, z in data}
        assert z_values & {1.0, -1.0}, (
            f"승패 샘플 없음 — z_values={z_values} (z 계산 버그 의심)"
        )

    def test_renju_compatible(self):
        """renju=True 환경에서도 병렬 self-play가 정상 완료."""
        torch.manual_seed(10)
        trainer = AlphaZeroTrainer(
            board_size=15, n_in_row=5, renju=True,
            n_simulations=5, n_iterations=1,
            self_play_games_per_iter=2,
            batch_size=32, device="cpu",
        )
        data = trainer._collect_self_play()
        assert len(data) >= 2 * 1 * 8, "renju 병렬 self-play 데이터 부족"

        valid_z = {-1.0, 0.0, 1.0}
        for _, pi, z in data:
            assert pi.shape == (225,), f"15×15 정책 shape 오류: {pi.shape}"
            assert z in valid_z,       f"z 값 오류: {z}"

    def test_no_episode_duplication(self):
        """증강을 제거한 원본 에피소드 수가 sp_games 게임과 일치하는지
        간접 확인: augment_data가 8배 증강하므로 총 샘플은 8의 배수여야 함."""
        SP = 4
        trainer = self._make_trainer(seed=4, sp=SP)
        data = trainer._collect_self_play()
        assert len(data) % 8 == 0, (
            f"총 샘플 수 {len(data)} 가 8의 배수가 아님 — 증강 누락/중복 의심"
        )
