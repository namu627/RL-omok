"""
env/test_renju_env.py

GomokuEnv(renju=True) 동작 검증:
  - 흑 금수(장목·44·33) → 즉시 패배
  - 흑 5목 → 승리 (금수 우선순위 무시)
  - 백 6목 이상 → 백 승리 (백은 제약 없음)
  - legal_actions()에서 흑의 금수 위치 제외
  - renju=False 기본값 동작 불변
"""

import numpy as np
import pytest
from env.gomoku import GomokuEnv, BLACK, WHITE, EMPTY


def renju_env():
    """15×15 렌주 환경, reset 완료 상태."""
    env = GomokuEnv(board_size=15, n_in_row=5, renju=True)
    env.reset()
    return env


# ─────────────────────────────────────────────────────────────────
# 금수 → 즉시 패배
# ─────────────────────────────────────────────────────────────────

class TestForbiddenMoveLoss:

    def test_overline_black_loses(self):
        """흑이 장목(6연속)을 두면 즉시 패배 / 백 승리."""
        env = renju_env()
        # 가로로 흑 5개 배치: cols 4~8, row 7
        for c in range(4, 9):
            env.board[7, c] = BLACK
        env.current_player = BLACK

        action = 7 * 15 + 3   # col 3 → 6연속 장목
        _, reward, done, info = env.step(action)

        assert done is True
        assert reward == -1.0
        assert env.winner == WHITE
        assert info["reason"] == "금수-장목"

    def test_44_black_loses(self):
        """흑이 44(두 방향 동시 4) 착수 → 즉시 패배."""
        env = renju_env()
        # 가로 3 (7,4)(7,5)(7,6) + 세로 3 (4,7)(5,7)(6,7)
        for c in [4, 5, 6]:
            env.board[7, c] = BLACK
        for r in [4, 5, 6]:
            env.board[r, 7] = BLACK
        env.current_player = BLACK

        action = 7 * 15 + 7   # (7,7) = 44 금수
        _, reward, done, info = env.step(action)

        assert done is True
        assert reward == -1.0
        assert env.winner == WHITE
        assert info["reason"] == "금수-44"

    def test_33_black_loses(self):
        """흑이 33(두 방향 동시 열린 3) 착수 → 즉시 패배."""
        env = renju_env()
        # 가로 열린 3: (7,5)(7,6), 세로 열린 3: (5,7)(6,7)
        for c in [5, 6]:
            env.board[7, c] = BLACK
        for r in [5, 6]:
            env.board[r, 7] = BLACK
        env.current_player = BLACK

        action = 7 * 15 + 7   # (7,7) = 33 금수
        _, reward, done, info = env.step(action)

        assert done is True
        assert reward == -1.0
        assert env.winner == WHITE
        assert info["reason"] == "금수-33"


# ─────────────────────────────────────────────────────────────────
# 5목 우선 원칙
# ─────────────────────────────────────────────────────────────────

class TestFiveWins:

    def test_five_in_a_row_black_wins(self):
        """흑이 정확히 5목 달성 → 승리 (금수 판정 없음)."""
        env = renju_env()
        for c in range(4, 8):   # 4개 사전 배치
            env.board[7, c] = BLACK
        env.current_player = BLACK

        action = 7 * 15 + 8    # 5번째 돌 → 5목
        _, reward, done, _ = env.step(action)

        assert done is True
        assert reward == 1.0
        assert env.winner == BLACK

    def test_white_six_in_a_row_wins(self):
        """백은 6목 이상도 승리 (렌주 제약 없음)."""
        env = renju_env()
        for c in range(3, 8):   # 5개 사전 배치
            env.board[7, c] = WHITE
        env.current_player = WHITE

        action = 7 * 15 + 8    # 6번째 돌
        _, reward, done, _ = env.step(action)

        assert done is True
        assert reward == 1.0
        assert env.winner == WHITE


# ─────────────────────────────────────────────────────────────────
# legal_actions() 금수 필터
# ─────────────────────────────────────────────────────────────────

class TestLegalActions:

    def test_44_excluded_from_legal_actions(self):
        """44 금수 위치는 흑의 legal_actions()에서 제외."""
        env = renju_env()
        for c in [4, 5, 6]:
            env.board[7, c] = BLACK
        for r in [4, 5, 6]:
            env.board[r, 7] = BLACK
        env.current_player = BLACK

        actions = env.legal_actions()
        forbidden = 7 * 15 + 7

        assert forbidden not in actions

    def test_33_excluded_from_legal_actions(self):
        """33 금수 위치는 흑의 legal_actions()에서 제외."""
        env = renju_env()
        for c in [5, 6]:
            env.board[7, c] = BLACK
        for r in [5, 6]:
            env.board[r, 7] = BLACK
        env.current_player = BLACK

        actions = env.legal_actions()
        forbidden = 7 * 15 + 7

        assert forbidden not in actions

    def test_white_turn_no_filter(self):
        """백 차례에는 금수 필터 없음 — 모든 빈 칸이 합법."""
        env = renju_env()
        for c in [4, 5, 6]:
            env.board[7, c] = BLACK
        for r in [4, 5, 6]:
            env.board[r, 7] = BLACK
        env.current_player = WHITE   # 백 차례

        actions = env.legal_actions()
        candidate = 7 * 15 + 7      # 흑에겐 44 금수지만 백은 무관

        assert candidate in actions


# ─────────────────────────────────────────────────────────────────
# 기존 자유룰 동작 불변
# ─────────────────────────────────────────────────────────────────

class TestFreeRuleUnchanged:

    def test_default_env_6x6(self):
        """renju=False 기본값: 6×6 4목 환경 정상 동작."""
        env = GomokuEnv()   # defaults: board_size=6, n_in_row=4, renju=False
        env.reset()
        for c in range(3):
            env.board[0, c] = BLACK
        env.current_player = BLACK
        env.done = False

        _, reward, done, _ = env.step(3)   # 4-in-a-row

        assert done is True
        assert reward == 1.0
        assert env.winner == BLACK

    def test_renju_false_no_forbidden_check(self):
        """renju=False이면 흑이 44 상황을 둬도 금수 처리 안 함."""
        env = GomokuEnv(board_size=15, n_in_row=5, renju=False)
        env.reset()
        for c in [4, 5, 6]:
            env.board[7, c] = BLACK
        for r in [4, 5, 6]:
            env.board[r, 7] = BLACK
        env.current_player = BLACK
        env.done = False

        _, reward, done, info = env.step(7 * 15 + 7)   # 44 위치지만 금수 아님

        assert "reason" not in info   # 금수 처리 없음
        assert done is False          # 게임 계속

    def test_renju_requires_size_15(self):
        """renju=True는 board_size=15만 허용."""
        with pytest.raises(ValueError):
            GomokuEnv(board_size=6, n_in_row=5, renju=True)
