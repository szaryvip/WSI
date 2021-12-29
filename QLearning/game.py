import pygame
from draw_board import FRAME_SIZE, draw_board, draw_image
from agent import Agent
from map_generator import prepare_correct_map
from typing import Tuple
from time import sleep


def game(start_point: Tuple[int, int], hole_proba: float,
         width: int = 8, height: int = 8):
    pygame.init()
    mymap = prepare_correct_map(start_point, hole_proba)
    size = max(len(mymap), len(mymap[0]))
    screen = pygame.display.set_mode([size*FRAME_SIZE,
                                      size*FRAME_SIZE])
    draw_board(screen, mymap)
    random_player = Agent(start_point, "random", mymap,
                          screen)
    q_uber = Agent(start_point, "random", mymap, screen)
    pygame.display.flip()
    end = False
    random_end = False
    while not end:
        sleep(1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = True
        q_uber.move()
        if q_uber.is_winner():
            print("Agent Q Uber dotarł do mety")
            end = True
            continue
        if q_uber.in_hole():
            print("Agent Q Uber trafił w stażystę!")
            q_uber.back_to(start_point)
        if not random_end:
            random_player.move()
        draw_image(screen, "img/mustang.jpeg", 1, 1)
        if random_player.is_winner():
            random_end = True
            print("Agent randomowy dotarł do mety")
        if random_player.in_hole():
            random_player.back_to(start_point)
            print("Agent randomowy trafił w stażystę!")
        pygame.display.flip()
    # to stop board before quit
    while end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                end = False
    pygame.quit()


if __name__ == "__main__":
    game((1, 1), 0.4)
