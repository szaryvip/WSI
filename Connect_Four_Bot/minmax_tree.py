import numpy as np


class Node:
    _value = 0
    _sons = []
    _is_terminal = False

    def __init__(self, board: np.array, depth: int):
        self._contain = board

    def get_value(self) -> int:
        return self._value

    def set_value(self, value: int) -> None:
        self._value = 0


class MMTree:
    def __init__(self, board: np.array, depth: int):
        self._root = Node(board)
        if depth > 0:
            self._depth = depth
        else:
            self._depth = 1

    def calculate_tree(self) -> None:
        pass

    def what_move(self, is_max: bool) -> int:
        pass
