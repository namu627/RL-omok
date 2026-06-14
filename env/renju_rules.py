"""
env/renju_rules.py

Renju 금수 규칙 구현.
인터페이스 정의 — 구현은 단계별로 채워 넣을 것.

좌표: 0-indexed (row, col). 보드: 15×15 int8 ndarray.
상수: EMPTY=0, BLACK=1, WHITE=2  (gomoku.py의 -1과 독립된 별도 표현)

단계별 구현 순서
  Phase 1: is_forbidden_overline, is_forbidden_44,
            is_forbidden_33_nonrecursive, classify_move
  Phase 2: is_forbidden_33_recursive
            (+ classify_move의 recursive_33=True 경로)
"""

import numpy as np

EMPTY: int = 0
BLACK: int = 1
WHITE: int = 2

BOARD_SIZE: int = 15

# 4방향: 가로, 세로, ↘ 대각, ↗ 대각
DIRECTIONS: list[tuple[int, int]] = [(0, 1), (1, 0), (1, 1), (1, -1)]


# ─────────────────────────────────────────────
# 내부 보조 함수
# ─────────────────────────────────────────────

def _count_dir(board: np.ndarray, row: int, col: int,
               dr: int, dc: int, color: int) -> int:
    """(row,col) 제외, (dr,dc) 방향으로 연속된 color 돌 수."""
    n, r, c = 0, row + dr, col + dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
        n += 1
        r += dr
        c += dc
    return n


def _line_len(board: np.ndarray, row: int, col: int,
              dr: int, dc: int, color: int) -> int:
    """(row,col) 포함, 해당 축 양방향 총 연속 길이."""
    return (1
            + _count_dir(board, row, col,  dr,  dc, color)
            + _count_dir(board, row, col, -dr, -dc, color))


def _count_fours_in_direction(board: np.ndarray, row: int, col: int,
                              dr: int, dc: int) -> int:
    """
    (board에 이미 (row,col)=BLACK 착수된 상태)
    해당 방향 축에서 (row,col)을 포함하는 독립된 "4" 패턴 수를 반환.

    "4" 패턴: 5-셀 윈도우 안에 흑 4개 + 빈칸 1개이며,
    빈칸을 채웠을 때 정확히 5연속이 되는 경우.
    같은 흑돌 4개를 공유하는 윈도우(예: 열린 4의 두 완성점)는 같은 4로 취급.
    """
    seen: set[frozenset] = set()
    count = 0
    for start_offset in range(5):
        r0 = row - start_offset * dr
        c0 = col - start_offset * dc
        vals: list[int] = []
        pos: list[tuple[int, int]] = []
        valid = True
        for i in range(5):
            r, c = r0 + i * dr, c0 + i * dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                vals.append(int(board[r][c]))
                pos.append((r, c))
            else:
                valid = False
                break
        if not valid:
            continue
        if vals.count(BLACK) == 4 and vals.count(EMPTY) == 1:
            ei = vals.index(EMPTY)
            er, ec = pos[ei]
            b2 = board.copy()
            b2[er][ec] = BLACK
            if _line_len(b2, er, ec, dr, dc, BLACK) == 5:
                black_key = frozenset(pos[i] for i in range(5) if vals[i] == BLACK)
                if black_key not in seen:
                    seen.add(black_key)
                    count += 1
    return count


def classify_move(
    board: np.ndarray,
    row: int,
    col: int,
    *,
    recursive_33: bool = False,
) -> str:
    """
    흑의 착수 (row, col) 를 분류한다. board는 착수 전 상태.

    우선순위: 5목승리 > 금수-장목 > 금수-44 > 금수-33 > 합법

    Parameters
    ----------
    board       : 15×15 int8 배열, EMPTY/BLACK/WHITE. 착수 전.
    row, col    : 0-indexed 착수 위치.
    recursive_33: True 면 1단계 재귀로 33 판정.

    Returns
    -------
    '5목승리' | '금수-장목' | '금수-44' | '금수-33' | '합법'
    """
    b = board.copy()
    b[row][col] = BLACK

    # 5목 우선 — 정확히 5연속이면 금수 판정 없이 승리
    for dr, dc in DIRECTIONS:
        if _line_len(b, row, col, dr, dc, BLACK) == 5:
            return "5목승리"

    # 장목
    if is_forbidden_overline(board, row, col):
        return "금수-장목"

    # 44
    if is_forbidden_44(board, row, col):
        return "금수-44"

    # 33
    check_33 = is_forbidden_33_recursive if recursive_33 else is_forbidden_33_nonrecursive
    if check_33(board, row, col):
        return "금수-33"

    return "합법"


def is_forbidden_overline(board: np.ndarray, row: int, col: int) -> bool:
    """(row, col)에 흑을 두면 장목(6목 이상)인가. board는 착수 전."""
    b = board.copy()
    b[row][col] = BLACK
    return any(_line_len(b, row, col, dr, dc, BLACK) >= 6 for dr, dc in DIRECTIONS)


def is_forbidden_44(board: np.ndarray, row: int, col: int) -> bool:
    """(row, col)에 흑을 두면 44(4를 둘 이상 동시 생성)인가. board는 착수 전."""
    b = board.copy()
    b[row][col] = BLACK
    total = sum(_count_fours_in_direction(b, row, col, dr, dc) for dr, dc in DIRECTIONS)
    return total >= 2


def _count_open_threes_in_direction(board: np.ndarray, row: int, col: int,
                                    dr: int, dc: int) -> int:
    """
    (board에 이미 (row,col)=BLACK 착수된 상태)
    해당 방향에서 (row,col)을 포함하는 독립된 "열린 3" 패턴 수.

    열린 3: 6-셀 윈도우 [E][?][?][?][?][E] 에서
    양끝이 EMPTY, 내부 4칸 중 정확히 3칸이 BLACK, 1칸이 EMPTY인 경우.
    같은 흑돌 집합을 공유하는 윈도우는 동일 열린 3.
    """
    seen: set[frozenset] = set()
    count = 0
    # (row,col)이 윈도우 내 위치 1,2,3,4 중 하나
    for pos_in_win in range(1, 5):
        r0 = row - pos_in_win * dr
        c0 = col - pos_in_win * dc
        cells: list[tuple[int, int, int]] = []  # (value, r, c)
        valid = True
        for i in range(6):
            r, c = r0 + i * dr, c0 + i * dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                cells.append((int(board[r][c]), r, c))
            else:
                valid = False
                break
        if not valid:
            continue
        # 양끝 빈칸 확인
        if cells[0][0] != EMPTY or cells[5][0] != EMPTY:
            continue
        # 내부 4칸: 흑 3개 + 빈칸 1개
        interior = cells[1:5]
        interior_vals = [v for v, _, _ in interior]
        if interior_vals.count(BLACK) != 3 or interior_vals.count(EMPTY) != 1:
            continue
        black_key = frozenset((r, c) for v, r, c in interior if v == BLACK)
        if black_key not in seen:
            seen.add(black_key)
            count += 1
    return count


def is_forbidden_33_nonrecursive(board: np.ndarray, row: int, col: int) -> bool:
    """
    (row, col)에 흑을 두면 33인가 — 비재귀 판정.

    열린 3이 2개 이상이면 무조건 금수.
    완성 지점의 금수 여부는 확인하지 않는다.
    board는 착수 전.
    """
    b = board.copy()
    b[row][col] = BLACK
    total = sum(
        _count_open_threes_in_direction(b, row, col, dr, dc)
        for dr, dc in DIRECTIONS
    )
    return total >= 2


def _count_open_threes_recursive_in_direction(
    board: np.ndarray, row: int, col: int, dr: int, dc: int
) -> int:
    """
    재귀 버전: 각 열린 3 패턴의 완성 지점이 모두 44/장목금수이면 해당 3을 제외.
    board에는 이미 (row,col)=BLACK이 놓여 있다.

    같은 black_key의 모든 완성 지점을 수집한 뒤, 하나라도 합법이면 열린 3으로 유지.
    """
    # black_key → 완성 지점(들) 매핑
    key_to_completions: dict[frozenset, list[tuple[int, int]]] = {}

    for pos_in_win in range(1, 5):
        r0 = row - pos_in_win * dr
        c0 = col - pos_in_win * dc
        cells: list[tuple[int, int, int]] = []
        valid = True
        for i in range(6):
            r, c = r0 + i * dr, c0 + i * dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                cells.append((int(board[r][c]), r, c))
            else:
                valid = False
                break
        if not valid:
            continue
        if cells[0][0] != EMPTY or cells[5][0] != EMPTY:
            continue
        interior = cells[1:5]
        interior_vals = [v for v, _, _ in interior]
        if interior_vals.count(BLACK) != 3 or interior_vals.count(EMPTY) != 1:
            continue
        black_key = frozenset((r, c) for v, r, c in interior if v == BLACK)
        ei = interior_vals.index(EMPTY)
        er, ec = interior[ei][1], interior[ei][2]
        key_to_completions.setdefault(black_key, []).append((er, ec))

    count = 0
    for completions in key_to_completions.values():
        # 하나라도 금수가 아닌 완성 지점이 있으면 진짜 열린 3
        any_legal = any(
            not is_forbidden_overline(board, er, ec)
            and not is_forbidden_44(board, er, ec)
            for er, ec in completions
        )
        if any_legal:
            count += 1
    return count


def is_forbidden_33_recursive(board: np.ndarray, row: int, col: int) -> bool:
    """
    (row, col)에 흑을 두면 33인가 — 1단계 전진 판정.

    ★ 구현 한계 (1.5단계) — 이름은 recursive이지만 실제 재귀 호출은 없음:
      - 열린 3의 완성 지점이 44금수 또는 장목금수인 경우만 해당 열린 3을 제외.
      - 완성 지점이 33금수인 경우는 미처리 → 합법으로 간주하여 열린 3을 유지.
      - 실제 대국에서 이 엣지 케이스는 극히 드물게 발생.
      - 완전 재귀(완성 지점 33 재귀 판정 + visited 사이클 방지)는 향후 과제.

    판정 원칙:
      열린 3의 모든 완성 지점이 44/장목금수일 때만 해당 열린 3을 제외.
      하나라도 합법(44/장목 아닌) 완성 지점이 있으면 열린 3으로 유지.
    board는 착수 전.
    """
    b = board.copy()
    b[row][col] = BLACK
    total = sum(
        _count_open_threes_recursive_in_direction(b, row, col, dr, dc)
        for dr, dc in DIRECTIONS
    )
    return total >= 2
