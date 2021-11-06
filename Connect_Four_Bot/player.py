import numpy as np


class Player:
    _type = "easy"
    _color = (255, 0, 0)
    _depth = 0
    _is_max = True

    def __init__(self, is_first, color, type="easy", depth=0):
        if type == "maxmin":
            self._depth = depth
        elif type != "easy":
            print("Wrong type of Bot. Game will use standard values")
        self._is_max = is_first
        self._color = color

    def get_color(self) -> tuple:
        return self._color

    def make_move(self, board: np.array) -> int:
        if self._type == "easy":
            possibilities = []
            for index, col in enumerate(board[0]):
                if col == 0.0:
                    possibilities.append(index)
            move = np.random.choice(possibilities)
            return move
        else:
            pass
