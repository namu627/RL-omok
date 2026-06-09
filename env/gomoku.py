import numpy as np

EMPTY = 0
BLACK = 1  # 선공
WHITE = -1  # 후공


class GomokuEnv:
    """
    6x6 보드, 4목 승리 오목 환경 (OpenAI Gym 스타일).

    상태: (board_size, board_size) numpy array. 1=흑, -1=백, 0=빈칸.
    행동: int, 0 ~ board_size²-1  (row*board_size + col)
    보상: 승 +1 / 패 -1 / 무 0 / 진행 중 0  (현재 플레이어 시점)
    """

    def __init__(self, board_size: int = 6, n_in_row: int = 4):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.board: np.ndarray = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player: int = BLACK
        self.done: bool = False
        self.winner: int = EMPTY
        self.move_count: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = BLACK
        self.done = False
        self.winner = EMPTY
        self.move_count = 0
        return self.board.copy()

    def step(self, action: int):
        """
        action: int  (row * board_size + col)
        returns: (next_state, reward, done, info)
          reward: 현재 플레이어(step 호출 시점) 기준
        """
        if self.done:
            raise RuntimeError("Game is already over. Call reset().")

        row, col = divmod(action, self.board_size)

        if self.board[row, col] != EMPTY:
            raise ValueError(f"Illegal action {action}: cell ({row},{col}) is not empty.")

        self.board[row, col] = self.current_player
        self.move_count += 1

        reward, self.done = self._check_terminal(row, col)

        info = {"winner": self.winner, "move_count": self.move_count}
        next_state = self.board.copy()

        # 턴 교체 (게임이 끝나도 current_player는 마지막 착수 플레이어 유지)
        if not self.done:
            self.current_player = -self.current_player

        return next_state, reward, self.done, info

    def legal_actions(self, state: np.ndarray = None) -> list[int]:
        """빈 칸의 행동 인덱스 목록 반환."""
        board = state if state is not None else self.board
        indices = np.argwhere(board == EMPTY)
        return [r * self.board_size + c for r, c in indices]

    def render(self, mode: str = "human") -> None:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        col_header = "  " + " ".join(str(c) for c in range(self.board_size))
        print(col_header)
        for r in range(self.board_size):
            row_str = " ".join(symbols[v] for v in self.board[r])
            print(f"{r} {row_str}")
        player_name = "X(Black)" if self.current_player == BLACK else "O(White)"
        print(f"Turn: {player_name}  Moves: {self.move_count}")
        if self.done:
            if self.winner == EMPTY:
                print("Result: Draw")
            else:
                w = "X(Black)" if self.winner == BLACK else "O(White)"
                print(f"Result: {w} wins!")
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_terminal(self, row: int, col: int):
        """착수 직후 호출. (reward, done) 반환."""
        player = self.board[row, col]

        if self._count_streak(row, col, player) >= self.n_in_row:
            self.winner = player
            # 착수한 플레이어가 이겼으므로 reward=+1
            return 1.0, True

        # 무승부: 빈 칸 없음
        if self.move_count == self.board_size ** 2:
            return 0.0, True

        return 0.0, False

    def _count_streak(self, row: int, col: int, player: int) -> int:
        """4방향 중 최대 연속 길이를 반환."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        best = 0
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            best = max(best, count)
        return best
