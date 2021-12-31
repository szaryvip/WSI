import pygame
from typing import Tuple
import numpy as np
from draw_board import HOLE_COLOR, clear_position, draw_image

actions = {0: "right", 1: "down", 2: "left", 3: "up"}
rev_actions = {"right": 0, "down": 1, "left": 2, "up": 3}


class Agent:
    _mode = "random"
    _position = (0, 0)
    _map = None
    _scene = None

    def __init__(self, position: Tuple[int, int],
                 mode: str, mymap: np.ndarray, scene: pygame.display,
                 discount: float, learn_rate: float, epsilon: float) -> None:
        self._position = position
        self._map = mymap
        self._scene = scene
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

    def possible_moves(self):
        moves = []
        if self._position[0]-1 >= 0:
            moves.append("left")
        if self._position[1]-1 >= 0:
            moves.append("up")
        if self._position[0]+1 < len(self._map[0]):
            moves.append("right")
        if self._position[1]+1 < len(self._map):
            moves.append("down")
        return moves

    def make_move(self, direction: str):
        if direction == "left":
            new_position = (self._position[0]-1, self._position[1])
        elif direction == "up":
            new_position = (self._position[0], self._position[1]-1)
        elif direction == "right":
            new_position = (self._position[0]+1, self._position[1])
        elif direction == "down":
            new_position = (self._position[0], self._position[1]+1)
        clear_position(self._scene, self._position[0],
                       self._position[1], (0, 0, 0))
        if self._mode == "random":
            draw_image(self._scene, "img/nissan.jpeg",
                       new_position[0], new_position[1])
        else:
            draw_image(self._scene, "img/mustang.jpeg",
                       new_position[0], new_position[1])
        self._position = new_position

    def maximazing(self, possible_moves):
        maxi = []
        for direction in possible_moves:
            if direction == "left":
                index = 2
                new_position = (self._position[0]-1, self._position[1])
            elif direction == "up":
                index = 3
                new_position = (self._position[0], self._position[1]-1)
            elif direction == "right":
                index = 0
                new_position = (self._position[0]+1, self._position[1])
            elif direction == "down":
                index = 1
                new_position = (self._position[0], self._position[1]+1)
            if (self.values[new_position[1], new_position[0],
                            index] >= self.values[self._position[1],
                                                  self._position[0], index]):
                maxi.append(direction)
        return maxi

    def policy(self, possible_moves):
        prob = []
        arg_max = np.argmax(self.values[self._position[1], self._position[0]])
        for move in possible_moves:
            proba = self._epsilon / len(possible_moves)
            if self.values[self._position[1], self._position[0],
                           rev_actions[move]] == arg_max:
                proba += ((1-self._epsilon) /
                          len(self.maximazing(possible_moves)))
            prob.append(proba)
        return prob

    def move(self):
        possible_moves = self.possible_moves()
        if self._mode == "random":
            direction = np.random.choice(possible_moves)
            self.make_move(direction)
        else:
            prob = self.policy(possible_moves)
            for i in range(len(prob)):
                prob[i] = prob[i] / sum(prob)
            direction = np.random.choice(possible_moves, p=prob)
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
        clear_position(self._scene, self._position[0],
                       self._position[1], HOLE_COLOR)
        self._position = new_point
        if self._mode == "random":
            draw_image(self._scene, "img/nissan.jpeg",
                       self._position[0], self._position[1])
        else:
            pass
