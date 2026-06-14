"""
tests/test_network.py
AlphaZeroNet 단위 테스트.

실행: python -m pytest tests/test_network.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from agents.alphazero.network import AlphaZeroNet, board_to_tensor
from env.gomoku import BLACK, WHITE


# ── 공용 픽스처 ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def net6():
    """6×6 보드용 네트워크 (결정적 결과를 위해 seed 고정)."""
    torch.manual_seed(42)
    return AlphaZeroNet(board_size=6, n_res_blocks=2, n_filters=32)


@pytest.fixture(scope="module")
def net15():
    """15×15 보드용 네트워크."""
    torch.manual_seed(42)
    return AlphaZeroNet(board_size=15, n_res_blocks=3, n_filters=64)


def make_board_6x6():
    """간단한 6×6 테스트 보드 (흑 두 개, 백 한 개)."""
    board = np.zeros((6, 6), dtype=np.int8)
    board[0, 0] = BLACK
    board[0, 1] = WHITE
    board[3, 3] = BLACK
    return board


def make_board_15x15():
    """간단한 15×15 테스트 보드."""
    board = np.zeros((15, 15), dtype=np.int8)
    board[7, 7] = BLACK
    board[7, 8] = WHITE
    return board


# ── predict_batch 일관성 테스트 ──────────────────────────────────────

class TestPredictBatch:

    def test_single_item_matches_predict(self, net6):
        """predict_batch(단일 상태) == predict(단일 상태)."""
        board = make_board_6x6()
        player = BLACK

        policy_seq, value_seq = net6.predict(board, player)
        policies_batch, values_batch = net6.predict_batch([board], [player])

        assert len(policies_batch) == 1
        assert len(values_batch) == 1
        np.testing.assert_allclose(
            policies_batch[0], policy_seq, atol=1e-5,
            err_msg="단일 배치 정책이 predict()와 다름"
        )
        assert abs(values_batch[0] - value_seq) < 1e-5, (
            f"단일 배치 가치 불일치: {values_batch[0]:.8f} vs {value_seq:.8f}"
        )

    def test_two_states_match_sequential(self, net6):
        """predict_batch([s1, s2]) 각각이 predict(s1), predict(s2)와 일치."""
        board1 = make_board_6x6()
        board2 = np.zeros((6, 6), dtype=np.int8)
        board2[2, 2] = BLACK
        board2[4, 4] = WHITE
        board2[1, 5] = BLACK

        states  = [(board1, BLACK), (board2, WHITE)]
        boards  = [s[0] for s in states]
        players = [s[1] for s in states]

        policies_batch, values_batch = net6.predict_batch(boards, players)

        for i, (board, player) in enumerate(states):
            p_seq, v_seq = net6.predict(board, player)
            np.testing.assert_allclose(
                policies_batch[i], p_seq, atol=1e-5,
                err_msg=f"상태 {i}: 배치 정책이 순차 predict()와 다름"
            )
            assert abs(values_batch[i] - v_seq) < 1e-5, (
                f"상태 {i}: 배치 가치 불일치 {values_batch[i]:.8f} vs {v_seq:.8f}"
            )

    def test_large_batch_matches_sequential(self, net15):
        """8개 상태 배치가 순차 predict 8회와 모두 일치 (15×15)."""
        rng = np.random.default_rng(7)
        boards, players = [], []
        for _ in range(8):
            b = np.zeros((15, 15), dtype=np.int8)
            idxs = rng.choice(225, size=10, replace=False)
            for k, idx in enumerate(idxs):
                b[idx // 15, idx % 15] = BLACK if k % 2 == 0 else WHITE
            boards.append(b)
            players.append(BLACK if rng.random() > 0.5 else WHITE)

        policies_batch, values_batch = net15.predict_batch(boards, players)

        assert len(policies_batch) == 8
        assert len(values_batch) == 8

        for i, (board, player) in enumerate(zip(boards, players)):
            p_seq, v_seq = net15.predict(board, player)
            np.testing.assert_allclose(
                policies_batch[i], p_seq, atol=1e-5,
                err_msg=f"상태 {i}: 배치와 순차 predict 정책 불일치"
            )
            assert abs(values_batch[i] - v_seq) < 1e-5, (
                f"상태 {i}: 배치와 순차 predict 가치 불일치"
            )

    def test_output_shapes(self, net6, net15):
        """반환 배열 크기 검증."""
        b6 = [make_board_6x6()] * 3
        p6 = [BLACK, WHITE, BLACK]
        policies, values = net6.predict_batch(b6, p6)
        assert len(policies) == 3
        assert len(values) == 3
        assert policies[0].shape == (36,), f"6×6 policy 크기 오류: {policies[0].shape}"

        b15 = [make_board_15x15()] * 2
        p15 = [BLACK, WHITE]
        policies15, values15 = net15.predict_batch(b15, p15)
        assert len(policies15) == 2
        assert policies15[0].shape == (225,), f"15×15 policy 크기 오류: {policies15[0].shape}"

    def test_policy_sums_to_one(self, net6):
        """각 정책 벡터의 합이 1 (확률 분포)."""
        boards = [make_board_6x6(), np.zeros((6, 6), dtype=np.int8)]
        players = [BLACK, WHITE]
        policies, _ = net6.predict_batch(boards, players)
        for i, p in enumerate(policies):
            assert abs(p.sum() - 1.0) < 1e-5, f"정책 {i} 합 = {p.sum():.8f} ≠ 1"

    def test_values_in_range(self, net6):
        """가치 출력이 [-1, 1] 범위 내."""
        boards = [make_board_6x6()] * 5
        players = [BLACK] * 5
        _, values = net6.predict_batch(boards, players)
        for v in values:
            assert -1.0 <= v <= 1.0, f"가치 범위 초과: {v}"

    def test_different_players_give_different_values(self, net6):
        """같은 보드라도 플레이어가 다르면 결과가 달라야 함 (대칭 아님)."""
        board = make_board_6x6()
        p_black, v_black = net6.predict(board, BLACK)
        p_white, v_white = net6.predict(board, WHITE)
        # 완전 동일이면 구현 오류 (ch2 턴 채널이 다르므로 달라야 함)
        assert not np.allclose(p_black, p_white, atol=1e-6), (
            "BLACK/WHITE 정책이 동일함 — board_to_tensor ch2 채널 버그 의심"
        )
