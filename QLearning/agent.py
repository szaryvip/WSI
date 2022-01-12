from typing import Tuple
import numpy as np

# right down left up -- 0,1,2,3


class Agent:
    _mode = "random"
    _position = (1, 1)
    _map = None

    def __init__(self, position: Tuple[int, int],
                 mode: str, mymap: np.ndarray,
                 discount: float, learn_rate: float,
                 epsilon: float) -> None:
        self._position = position
        self._map = mymap
        if mode in ["random", "qlearning"]:
            self._mode = mode
        else:
            raise ValueError("Possible mode of agent is\
                radnom or qlearning")
        if mode == "qlearning":
            self._disc = discount
            self._learn_rate = learn_rate
            self._epsilon = epsilon
            self.values = np.zeros([len(mymap), len(mymap[0]), 4])

    def make_move(self, direction: str):
        if direction == 2:
            new_position = (self._position[0]-1, self._position[1])
        elif direction == 3:
            new_position = (self._position[0], self._position[1]-1)
        elif direction == 0:
            new_position = (self._position[0]+1, self._position[1])
        elif direction == 1:
            new_position = (self._position[0], self._position[1]+1)
        self._position = new_position

    def move(self):
        possible_moves = [0, 1, 2, 3]
        if self._mode == "random":
            direction = np.random.choice(possible_moves)
            self.make_move(direction)
        else:
            if np.random.rand() < self._epsilon:
                mov = None
                direction = -1000000
                for pos_move in possible_moves:
                    if direction < self.values[self._position[1],
                                               self._position[0],
                                               pos_move]:
                        direction = (self.values[self._position[1],
                                                 self._position[0],
                                                 pos_move])
                        mov = pos_move
                direction = mov
            else:
                direction = np.random.choice(possible_moves)
            self.make_move(direction)
        return direction

    def is_winner(self):
        if self._map[self._position[1], self._position[0]] == 200:
            return True
        return False

    def in_hole(self):
        if self._map[self._position[1], self._position[0]] == -1000:
            return True
        return False

    def get_position(self):
        return self._position

    def back_to(self, new_point: Tuple[int, int]):
        self._position = new_point
