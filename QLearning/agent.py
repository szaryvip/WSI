import pygame
from typing import Tuple
import numpy as np
from draw_board import clear_position, draw_image


class Agent:
    _mode = "random"
    _position = (0, 0)
    _map = None
    _scene = None

    def __init__(self, position: Tuple[int, int],
                 mode: str, mymap: np.ndarray, scene: pygame.display) -> None:
        self._position = position
        self._map = mymap
        self._scene = scene
        if mode in ["random", "qlearning"]:
            self._mode = mode
        else:
            raise ValueError("Possible mode of agent is\
                radnom or qlearning")

    def possible_moves(self):
        moves = []
        if self._position[0]-1 >= 0:
            move = (self._position[0]-1, self._position[1])
            if self._map[move[1], move[0]] != -1:
                moves.append("left")
        if self._position[1]-1 >= 0:
            move = (self._position[0], self._position[1]-1)
            if self._map[move[1], move[0]] != -1:
                moves.append("up")
        if self._position[0]+1 < len(self._map[0]):
            move = (self._position[0]+1, self._position[1])
            if self._map[move[1], move[0]] != -1:
                moves.append("right")
        if self._position[1]+1 < len(self._map):
            move = (self._position[0], self._position[1]+1)
            if self._map[move[1], move[0]] != -1:
                moves.append("down")
        return moves

    def move(self):
        possible_moves = self.possible_moves()
        if self._mode == "random":
            direction = np.random.choice(possible_moves)
            if direction == "left":
                new_position = (self._position[0]-1, self._position[1])
            elif direction == "up":
                new_position = (self._position[0], self._position[1]-1)
            elif direction == "right":
                new_position = (self._position[0]+1, self._position[1])
            elif direction == "down":
                new_position = (self._position[0], self._position[1]+1)
            clear_position(self._scene, self._position[0], self._position[1])
            draw_image(self._scene, "img/nissan.jpeg",
                       new_position[0], new_position[1])
            self._position = new_position
        else:
            pass

    def is_winner(self):
        if self._map[self._position[1], self._position[0]] == 2:
            return True
        return False
