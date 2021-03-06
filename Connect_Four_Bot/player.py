import numpy as np
from minmax_tree import minmaxalg


class Player:
    _type = "easy"
    _color = (255, 0, 0)
    _depth = 0
    _is_max = True

    def __init__(self, is_first, color, type="easy", depth=0):
        """
        Contructor of Player object
        Parameters:
        - is_first [bool] - if this player is first
        - color [tuple[int,int,int]] - color of player moves
        - type ["easy"/"minmax"] - type of bot
        - depth [unsigned int] - how many moves bot will calculate if maxmin"""
        if type == "minmax":
            self._type = "minmax"
            self._depth = depth
        elif type != "easy":
            print("Wrong type of Bot. Game will use standard values")
        self._is_max = is_first
        self._color = color

    def get_color(self) -> tuple:
        """
        Return color of bot moves"""
        return self._color

    def is_max(self) -> bool:
        """
        Return True if player is max player"""
        return self._is_max

    def make_move(self, board: np.array) -> int:
        """
        Choose column to make move
        Parameter;
        - board [np.array] - array with moves"""
        if self._type == "easy":
            possibilities = []
            for index, col in enumerate(board[0]):
                if col == 0.0:
                    possibilities.append(index)
            move = np.random.choice(possibilities)
            return move
        else:
            score, move = minmaxalg(board, None, self._depth,
                                    -10000, 10000, self._is_max)
            print(score)
            return move[0]
