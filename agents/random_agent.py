import random


class RandomAgent:
    """legal_actions 중 무작위로 선택하는 기준 에이전트."""

    def select_action(self, state, legal_actions: list[int]) -> int:
        return random.choice(legal_actions)
