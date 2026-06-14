"""
env/test_renju_rules.py

Renju 금수 규칙 테스트 정답지 (Phase 4a).
모든 케이스는 설계·검수 과정에서 좌표·라벨·근거가 확인되었음.

좌표 표기: 1-indexed (row, col), row 1 = 최상단, col 1 = 최좌단.
내부 변환: p(r, c) → 0-indexed (r-1, c-1).

Phase 1 (비재귀): TestOverline, TestDoubleFour, TestLegalMoves,
                  TestDoubleThreeNonRecursive
Phase 2 (1단계 재귀): TestDoubleThreeRecursive
"""

import numpy as np
import pytest

from env.renju_rules import (
    classify_move,
    is_forbidden_overline,
    is_forbidden_44,
    is_forbidden_33_nonrecursive,
    is_forbidden_33_recursive,
    EMPTY, BLACK, WHITE,
)


# ─────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────

def make_board(black=(), white=()):
    """15×15 빈 보드에 돌 배치. 좌표는 1-indexed (row, col)."""
    board = np.zeros((15, 15), dtype=np.int8)
    for r, c in black:
        board[r - 1][c - 1] = BLACK
    for r, c in white:
        board[r - 1][c - 1] = WHITE
    return board


def p(r, c):
    """1-indexed (row, col) → 0-indexed tuple."""
    return r - 1, c - 1


def board_after_move(black, white, move):
    """
    착수(move)까지 포함된 보드 반환.
    재귀 케이스의 완성 지점 중간 판정에 사용.
    """
    board = make_board(black, white)
    r, c = move
    board[r - 1][c - 1] = BLACK
    return board


# ─────────────────────────────────────────────────────────────────
# Phase 1 ― 장목
# ─────────────────────────────────────────────────────────────────

class TestOverline:
    """장목: 흑이 6목 이상 만드는 수 → 금수-장목."""

    def test_chanmok_1_horizontal(self):
        """가로 6목 (col 3~8)."""
        board = make_board(black=[(7,4),(7,5),(7,6),(7,7),(7,8)])
        assert classify_move(board, *p(7,3)) == "금수-장목"

    def test_chanmok_2_vertical_gap(self):
        """세로 6목, 중간 (6,7) 연결로 row 3~8 완성."""
        board = make_board(black=[(3,7),(4,7),(5,7),(7,7),(8,7)])
        assert classify_move(board, *p(6,7)) == "금수-장목"

    def test_chanmok_3_diagonal(self):
        """↗ 대각 6목 (row+col=11 라인, (5,6) 연결)."""
        board = make_board(black=[(8,3),(7,4),(6,5),(4,7),(3,8)])
        assert classify_move(board, *p(5,6)) == "금수-장목"


# ─────────────────────────────────────────────────────────────────
# Phase 1 ― 44
# ─────────────────────────────────────────────────────────────────

class TestDoubleFour:
    """44: 한 수로 4를 둘 이상 동시 생성 → 금수-44."""

    def test_44_1_horizontal_vertical(self):
        """열린 4: 가로 + 세로."""
        board = make_board(black=[(5,7),(6,7),(7,7),(8,4),(8,5),(8,6)])
        assert classify_move(board, *p(8,7)) == "금수-44"

    def test_44_2_diagonal_horizontal(self):
        """열린 4: ↘ 대각 + 가로."""
        board = make_board(black=[(5,5),(6,6),(8,8),(7,4),(7,5),(7,6)])
        assert classify_move(board, *p(7,7)) == "금수-44"

    def test_44_3_same_line_two_fours(self):
        """한 직선에 띈 4 두 개: B B B _ * _ B B B."""
        board = make_board(black=[(8,3),(8,4),(8,5),(8,9),(8,10),(8,11)])
        assert classify_move(board, *p(8,7)) == "금수-44"

    def test_44_4_closed_four_horizontal_vertical(self):
        """닫힌 4 × 2: 가로(백 왼쪽 차단) + 세로(백 아래 차단)."""
        board = make_board(
            black=[(5,7),(6,7),(7,7),(8,4),(8,5),(8,6)],
            white=[(8,3),(9,7)],
        )
        assert classify_move(board, *p(8,7)) == "금수-44"

    def test_44_5_closed_four_diagonal_horizontal(self):
        """닫힌 4 × 2: ↘ 대각(백 위 차단) + 가로(백 오른쪽 차단)."""
        board = make_board(
            black=[(5,5),(6,6),(7,7),(8,5),(8,6),(8,7)],
            white=[(4,4),(8,9)],
        )
        assert classify_move(board, *p(8,8)) == "금수-44"

    def test_44_6_skip_four_bb_bb(self):
        """띈 4(BB_BB형) + 연속 4: 띈 4도 4로 카운트됨을 검증."""
        board = make_board(black=[(7,3),(7,4),(7,6),(4,7),(5,7),(6,7)])
        assert classify_move(board, *p(7,7)) == "금수-44"


# ─────────────────────────────────────────────────────────────────
# Phase 1 ― 정상 합법수
# ─────────────────────────────────────────────────────────────────

class TestLegalMoves:
    """금수 조건에 해당하지 않는 정상 수."""

    def test_legal_single_four(self):
        """4 하나만 생성 → 합법."""
        board = make_board(black=[(7,5),(7,6),(7,7)])
        assert classify_move(board, *p(7,8)) == "합법"

    def test_legal_single_open_three(self):
        """열린 3 하나만 생성 → 합법."""
        board = make_board(black=[(7,6),(7,7)])
        assert classify_move(board, *p(7,8)) == "합법"

    def test_five_in_a_row_win(self):
        """정확히 5목 → 5목승리 (장목 아님)."""
        board = make_board(black=[(7,4),(7,5),(7,6),(7,7)])
        assert classify_move(board, *p(7,8)) == "5목승리"


# ─────────────────────────────────────────────────────────────────
# Phase 1 ― 비재귀 33
# ─────────────────────────────────────────────────────────────────

class TestDoubleThreeNonRecursive:
    """비재귀 33: 열린 3이 2개 이상이면 무조건 금수."""

    # ── 금수 케이스: is_forbidden_33_nonrecursive ──

    def test_33_1_horizontal_vertical(self):
        """가로 연속 3 + 세로 연속 3 → 33금수."""
        board = make_board(black=[(5,7),(6,7),(7,5),(7,6)])
        assert is_forbidden_33_nonrecursive(board, *p(7,7)) is True

    def test_33_2_horizontal_skip_diagonal(self):
        """가로 연속 3 + ↘ 대각 띈 3(BB_B형) → 33금수."""
        board = make_board(black=[(5,5),(6,6),(8,6),(8,7)])
        assert is_forbidden_33_nonrecursive(board, *p(8,8)) is True

    def test_33_3_two_diagonals(self):
        """↘ 대각 연속 3 + ↗ 대각 연속 3 → 33금수."""
        board = make_board(black=[(5,5),(6,6),(5,9),(6,8)])
        assert is_forbidden_33_nonrecursive(board, *p(7,7)) is True

    # ── 합법 케이스: is_forbidden_33_nonrecursive ──

    def test_not_33_closed_three_by_white(self):
        """열린 3 + 닫힌 3(백돌로 막힘) → 닫힌 3 미계산 → 합법.
        버그: 닫힌 3을 열린 3으로 오산하면 33금수 오판."""
        board = make_board(black=[(5,7),(6,7),(7,5),(7,6)], white=[(8,7)])
        assert is_forbidden_33_nonrecursive(board, *p(7,7)) is False

    def test_not_33_closed_three_by_edge(self):
        """열린 3 + 닫힌 3(보드 끝으로 막힘) → 합법."""
        board = make_board(black=[(1,5),(1,6),(2,7),(3,7)])
        assert is_forbidden_33_nonrecursive(board, *p(1,7)) is False

    def test_not_33_single_open_three(self):
        """열린 3 하나만 → 합법."""
        board = make_board(black=[(7,6),(7,7)])
        assert is_forbidden_33_nonrecursive(board, *p(7,8)) is False

    def test_not_33_closed_four_plus_open_three(self):
        """닫힌 4 + ↗ 열린 3 → 4는 열린 3으로 미계산 → 합법.
        버그: 닫힌 4를 열린 3으로 오산하면(한쪽 끝만 확인) 33금수 오판."""
        board = make_board(
            black=[(7,4),(7,5),(7,6),(8,6),(9,5)],
            white=[(7,3)],
        )
        assert is_forbidden_33_nonrecursive(board, *p(7,7)) is False

    # ── classify_move 통합 검증 ──

    def test_classify_33_1(self):
        board = make_board(black=[(5,7),(6,7),(7,5),(7,6)])
        assert classify_move(board, *p(7,7)) == "금수-33"

    def test_classify_33_2(self):
        board = make_board(black=[(5,5),(6,6),(8,6),(8,7)])
        assert classify_move(board, *p(8,8)) == "금수-33"

    def test_classify_33_3(self):
        board = make_board(black=[(5,5),(6,6),(5,9),(6,8)])
        assert classify_move(board, *p(7,7)) == "금수-33"

    def test_classify_not_33_closed_white(self):
        board = make_board(black=[(5,7),(6,7),(7,5),(7,6)], white=[(8,7)])
        assert classify_move(board, *p(7,7)) == "합법"

    def test_classify_not_33_closed_edge(self):
        board = make_board(black=[(1,5),(1,6),(2,7),(3,7)])
        assert classify_move(board, *p(1,7)) == "합법"

    def test_classify_not_33_single_three(self):
        board = make_board(black=[(7,6),(7,7)])
        assert classify_move(board, *p(7,8)) == "합법"

    def test_classify_not_33_closed_four(self):
        board = make_board(
            black=[(7,4),(7,5),(7,6),(8,6),(9,5)],
            white=[(7,3)],
        )
        assert classify_move(board, *p(7,7)) == "합법"


# ─────────────────────────────────────────────────────────────────
# Phase 2 ― 1단계 재귀 33
# ─────────────────────────────────────────────────────────────────
#
# R-1: 세로 열린 3의 완성 지점 (4,7)·(8,7) 모두 44금수
#      → 세로 열린 3 제외 → 진짜 열린 3 1개(가로) → 합법
#
# R-2: 세로 열린 3의 완성 지점 (4,7)=44금수, (8,7)=합법
#      → (8,7)이 합법이므로 세로 열린 3 유지 → 진짜 열린 3 2개 → 33금수

_R1_BLACK = [
    (4,4),(4,5),(4,6),      # (4,7) 완성 시 가로 4 → 44
    (5,7),(6,7),             # 세로 열린 3 위쪽
    (7,5),(7,6),             # 가로 열린 3 왼쪽
    (8,4),(8,5),(8,6),      # (8,7) 완성 시 가로 4 → 44
]

_R2_BLACK = [
    (4,4),(4,5),(4,6),      # (4,7) 완성 시 가로 4 → 44
    (5,7),(6,7),             # 세로 열린 3 위쪽
    (7,5),(7,6),             # 가로 열린 3 왼쪽
    # (8,4)(8,5)(8,6) 없음 → (8,7) 합법
]


class TestDoubleThreeRecursive:
    """
    1단계 재귀 33.
    원칙: 열린 3의 모든 완성 지점이 금수일 때만 제외.
          하나라도 합법인 완성 지점이 있으면 열린 3 유지.
    """

    # ── R-1 중간 판정 (착수 (7,7) 이후 보드 기준) ──

    def test_r1_completion_4_7_is_44(self):
        """R-1: 세로 열린 3 완성 지점 (4,7) → 44금수.
        수직 4 (4,7)~(7,7) + 수평 4 (4,4)~(4,7) 동시 생성."""
        board = board_after_move(_R1_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(4,7)) is True

    def test_r1_completion_8_7_is_44(self):
        """R-1: 세로 열린 3 완성 지점 (8,7) → 44금수.
        수직 4 (5,7)~(8,7) + 수평 4 (8,4)~(8,7) 동시 생성."""
        board = board_after_move(_R1_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(8,7)) is True

    def test_r1_completion_7_4_is_legal(self):
        """R-1: 가로 열린 3 완성 지점 (7,4) → 합법 (가로 4 하나뿐)."""
        board = board_after_move(_R1_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(7,4)) is False
        assert is_forbidden_overline(board, *p(7,4)) is False

    def test_r1_completion_7_8_is_legal(self):
        """R-1: 가로 열린 3 완성 지점 (7,8) → 합법 (가로 4 하나뿐)."""
        board = board_after_move(_R1_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(7,8)) is False
        assert is_forbidden_overline(board, *p(7,8)) is False

    # ── R-1 최종 판정 ──

    def test_r1_nonrecursive_sees_33(self):
        """R-1: 비재귀는 열린 3 후보 2개 → 33금수로 오판."""
        board = make_board(black=_R1_BLACK)
        assert is_forbidden_33_nonrecursive(board, *p(7,7)) is True

    def test_r1_recursive_is_legal(self):
        """R-1: 재귀 판정 → 합법.
        세로 열린 3의 완성 지점 (4,7)·(8,7) 전부 44금수 → 세로 열린 3 제외
        → 진짜 열린 3 = 가로 1개 → 33 불성립."""
        board = make_board(black=_R1_BLACK)
        assert is_forbidden_33_recursive(board, *p(7,7)) is False

    def test_r1_classify_recursive(self):
        """R-1: classify_move(recursive_33=True) → 합법."""
        board = make_board(black=_R1_BLACK)
        assert classify_move(board, *p(7,7), recursive_33=True) == "합법"

    # ── R-2 중간 판정 (착수 (7,7) 이후 보드 기준) ──

    def test_r2_completion_4_7_is_44(self):
        """R-2: 세로 열린 3 완성 지점 (4,7) → 44금수 (R-1과 동일)."""
        board = board_after_move(_R2_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(4,7)) is True

    def test_r2_completion_8_7_is_legal(self):
        """R-2: 세로 열린 3 완성 지점 (8,7) → 합법.
        row 8에 가로 흑 없으므로 수직 4 하나뿐 → 44 아님.
        이 한 합법 완성 지점이 세로 열린 3을 살림."""
        board = board_after_move(_R2_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(8,7)) is False
        assert is_forbidden_overline(board, *p(8,7)) is False

    def test_r2_completion_7_4_is_legal(self):
        """R-2: 가로 열린 3 완성 지점 (7,4) → 합법."""
        board = board_after_move(_R2_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(7,4)) is False

    def test_r2_completion_7_8_is_legal(self):
        """R-2: 가로 열린 3 완성 지점 (7,8) → 합법."""
        board = board_after_move(_R2_BLACK, [], (7,7))
        assert is_forbidden_44(board, *p(7,8)) is False

    # ── R-2 최종 판정 ──

    def test_r2_nonrecursive_sees_33(self):
        """R-2: 비재귀도 33금수."""
        board = make_board(black=_R2_BLACK)
        assert is_forbidden_33_nonrecursive(board, *p(7,7)) is True

    def test_r2_recursive_is_still_33(self):
        """R-2: 재귀 판정도 33금수.
        (8,7) 합법 → 세로 열린 3 유지 → 진짜 열린 3 2개 → 33 성립.
        함정: (4,7) 금수 하나만 보고 세로 열린 3 전체를 성급히 제외하면
              진짜 열린 3이 1개로 줄어 합법으로 오판."""
        board = make_board(black=_R2_BLACK)
        assert is_forbidden_33_recursive(board, *p(7,7)) is True

    def test_r2_classify_recursive(self):
        """R-2: classify_move(recursive_33=True) → 금수-33."""
        board = make_board(black=_R2_BLACK)
        assert classify_move(board, *p(7,7), recursive_33=True) == "금수-33"
