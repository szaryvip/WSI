import pygame
from draw_board import FRAME_SIZE, draw_board
from agent import Agent
from map_generator import prepare_correct_map
from typing import Tuple
import numpy as np
from time import sleep

actions = {"right": 0, "down": 1, "left": 2, "up": 3}


def game(start_point: Tuple[int, int], hole_proba: float,
         max_iter: int, discount: float = 1.0,
         learn_rate: float = 0.2, epsilon: float = 0.9,
         width: int = 8, height: int = 8):
    pygame.init()
    mymap = prepare_correct_map(start_point, hole_proba)
    size = max(len(mymap), len(mymap[0]))
    screen = pygame.display.set_mode([size*FRAME_SIZE,
                                      size*FRAME_SIZE])
    draw_board(screen, mymap)
    random_player = Agent(start_point, "random", mymap,
                          screen, 0, 0, 0)
    q_uber = Agent(start_point, "qlearning", mymap, screen,
                   discount, learn_rate, epsilon)
    pygame.display.flip()
    end = False
    random_end = False
    iteration = 0
    while not end:
        sleep(0.5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = True
        q_pos_old = q_uber.get_position()
        q_move = q_uber.move()
        q_pos = q_uber.get_position()
        reward = mymap[q_pos[1], q_pos[0]]
        action_index = actions[q_move]
        old_value = q_uber.values[q_pos_old[1], q_pos_old[0], action_index]
        diff = learn_rate * (reward +
                             (discount *
                              np.max(q_uber.values[q_pos[1],
                                                   q_pos[0]])) -
                             old_value)
        q_uber.values[q_pos_old[1],
                      q_pos_old[0], action_index] = old_value + diff
        print(q_uber.values[q_pos_old[1], q_pos_old[0]])
        sleep(2)
        if q_uber.is_winner():
            print("Agent Q Uber dotarł do mety")
            pygame.display.flip()
            end = True
            continue
        if q_uber.in_hole():
            print("Agent Q Uber trafił w stażystę!")
            q_uber.back_to(start_point)
        sleep(0.5)
        if not random_end:
            random_player.move()
        if random_player.is_winner():
            random_end = True
            print("Agent randomowy dotarł do mety")
        if random_player.in_hole():
            random_player.back_to(start_point)
            print("Agent randomowy trafił w stażystę!")
        pygame.display.flip()
        iteration += 1
        if iteration == max_iter:
            end = True
    # to stop board before quit
    while end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = False
    pygame.quit()


if __name__ == "__main__":
    game((1, 1), 0.3, 100)
