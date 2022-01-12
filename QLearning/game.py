import pygame
from draw_board import FRAME_SIZE, draw_board, clear_position, draw_image
from agent import Agent
from map_generator import prepare_correct_map
from typing import Tuple
import numpy as np
from time import sleep


class Game:
    _start_point = None
    _my_map = None
    _screen = None
    _q_uber = None
    _random = None
    _learning_rate = None
    _discount = None
    _iterations = None

    def __init__(self, start_point: Tuple[int, int], hole_proba: int,
                 max_iter: int, discount: float = 1.0, learn_rate: float = 0.8,
                 epsilon: float = 0.9, width: int = 8, height: int = 8):
        self._discount = discount
        self._learning_rate = learn_rate
        self._iterations = max_iter
        self._my_map = prepare_correct_map(start_point, hole_proba,
                                           width, height)
        # self._my_map = np.array([[-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000],
        #                         [-1000, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1000],
        #                         [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1, -1000],
        #                         [-1000, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1, -1, -1, -1, -1, -1, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1000, -1000, -1000, -1000, -1000, -1, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1000, -1, -1, 200, -1000, -1, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1000, -1, -1000, -1000, -1000, -1, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1000, -1, -1, -1, -1, -1, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1000, -1, -1000],
        #                         [-1000, -1, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1, -1000],
        #                         [-1000, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1000],
        #                         [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]])
        size = max(width+2, height+2)
        self._screen = pygame.display.set_mode([size*FRAME_SIZE,
                                                size*FRAME_SIZE])
        self._q_uber = Agent(start_point, 'qlearning', self._my_map,
                             discount, learn_rate, epsilon)
        self._random = Agent(start_point, 'random', self._my_map, 0, 0, 0)

    def set_start_point(self, start_point: Tuple[int, int]):
        self._start_point = start_point

    def play(self):
        self._q_uber.back_to(self._start_point)
        self._random.back_to(self._start_point)
        end = False
        random_end = False
        iteration = 0
        while not end:
            q_pos_old = self._q_uber.get_position()
            q_move = self._q_uber.move()
            q_pos = self._q_uber.get_position()
            reward = self._my_map[q_pos[1], q_pos[0]]
            action_index = q_move
            old_value = self._q_uber.values[q_pos_old[1],
                                            q_pos_old[0], action_index]
            diff = self._learning_rate * (reward +
                                          (self._discount *
                                           np.max(self._q_uber.values[q_pos[1],
                                                  q_pos[0]])) - old_value)
            self._q_uber.values[q_pos_old[1],
                                q_pos_old[0], action_index] = old_value + diff
            if self._q_uber.is_winner():
                # print("Agent Q Uber dotarł do mety")
                end = True
                continue
            if self._q_uber.in_hole():
                # print("Agent Q Uber trafił w stażystę!")
                self._q_uber.back_to(self._start_point)
                q_pos = self._q_uber.get_position()
            if not random_end:
                self._random.move()
                if self._random.is_winner():
                    random_end = True
                    # print("Agent randomowy dotarł do mety")
                if self._random.in_hole():
                    self._random.back_to(self._start_point)
            iteration += 1
            if iteration == self._iterations:
                end = True
        return iteration

    def show_game(self):
        self._q_uber.back_to(self._start_point)
        self._random.back_to(self._start_point)
        draw_board(self._screen, self._my_map, self._start_point)
        pygame.display.flip()
        end = False
        random_end = False
        iteration = 0
        sleep(2)
        print(self._my_map)
        while not end:
            sleep(0.1)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    end = True
            q_pos_old = self._q_uber.get_position()
            q_move = self._q_uber.move()
            q_pos = self._q_uber.get_position()
            reward = self._my_map[q_pos[1], q_pos[0]]
            action_index = q_move
            old_value = self._q_uber.values[q_pos_old[1],
                                            q_pos_old[0], action_index]
            diff = self._learning_rate * (reward +
                                          (self._discount *
                                           np.max(self._q_uber.values[q_pos[1],
                                                  q_pos[0]])) - old_value)
            clear_position(self._screen, q_pos_old[0], q_pos_old[1],
                           (0, 0, 0))
            draw_image(self._screen, 'img/mustang.jpeg', q_pos[0], q_pos[1])
            pygame.display.flip()
            self._q_uber.values[q_pos_old[1],
                                q_pos_old[0], action_index] = old_value + diff
            sleep(0.1)
            if self._q_uber.is_winner():
                # print("Agent Q Uber dotarł do mety")
                pygame.display.flip()
                end = True
                continue
            if self._q_uber.in_hole():
                # print("Agent Q Uber trafił w stażystę!")
                clear_position(self._screen, q_pos[0], q_pos[1], (255, 0, 0))
                self._q_uber.back_to(self._start_point)
                q_pos = self._q_uber.get_position()
                draw_image(self._screen, 'img/mustang.jpeg', q_pos[0],
                           q_pos[1])
                pygame.display.flip()
                sleep(0.1)
            if not random_end:
                old_pos = self._random.get_position()
                self._random.move()
                new_pos = self._random.get_position()
                clear_position(self._screen, old_pos[0], old_pos[1],
                               (0, 0, 0))
                draw_image(self._screen, 'img/nissan.jpeg', new_pos[0],
                           new_pos[1])
                if self._random.is_winner():
                    random_end = True
                    # print("Agent randomowy dotarł do mety")
                if self._random.in_hole():
                    clear_position(self._screen, new_pos[0], new_pos[1],
                                   (255, 0, 0))
                    self._random.back_to(self._start_point)
                    new_pos = self._random.get_position()
                    draw_image(self._screen, 'img/nissan.jpeg', new_pos[0],
                               new_pos[1])
                    # print("Agent randomowy trafił w stażystę!")
                pygame.display.flip()
            iteration += 1
            if iteration == self._iterations:
                end = True
        # to stop board before quit
        while end:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    end = False
        return iteration


if __name__ == "__main__":
    pygame.init()
    game = Game((1, 1), 0.4, 200, epsilon=1, width=13, height=13)
    for _ in range(200):
        correct_start = False
        while not correct_start:
            start_x = np.random.randint(1, 14)
            start_y = np.random.randint(1, 14)
            if game._my_map[start_y, start_x] == -1:
                correct_start = True
        game.set_start_point((start_x, start_y))
        game.play()
    game.set_start_point((1, 1))
    game.show_game()
    pygame.quit()
