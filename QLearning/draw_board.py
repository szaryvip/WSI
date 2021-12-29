import pygame
from map_generator import prepare_correct_map
import numpy as np
from typing import Tuple

FRAME_COLOR = (0, 0, 255)
HOLE_COLOR = (255, 0, 0)
FRAME_SIZE = 66


def draw_frames(screen: pygame.display, mymap: np.ndarray):
    for y in range(len(mymap)):
        for x in range(len(mymap[0])):
            pygame.draw.rect(screen, FRAME_COLOR,
                             pygame.Rect(x*FRAME_SIZE, y*FRAME_SIZE,
                                         FRAME_SIZE, FRAME_SIZE), 1)


def draw_image(screen: pygame.display,
               path: str, x: int, y: int):
    image = pygame.image.load(path)
    image = pygame.transform.scale(image, (FRAME_SIZE-2, FRAME_SIZE-2))
    screen.blit(image, (x*FRAME_SIZE+1, y*FRAME_SIZE+1))


def draw_board(screen: pygame.display, mymap: np.ndarray):
    draw_frames(screen, mymap)
    for y in range(len(mymap)):
        for x in range(len(mymap[0])):
            if mymap[y, x] == 1:
                draw_image(screen, 'img/nissan.jpeg', x, y)
                draw_image(screen, 'img/mustang.jpeg', x, y)
            if mymap[y, x] == 2:
                draw_image(screen, 'img/finish_flag.jpg', x, y)
            if mymap[y, x] == -1:
                pygame.draw.rect(screen, HOLE_COLOR,
                                 pygame.Rect(x*FRAME_SIZE+1, y*FRAME_SIZE+1,
                                             FRAME_SIZE-2, FRAME_SIZE-2))


def clear_position(screen: pygame.display, x: int, y: int,
                   color: Tuple[int, int, int]):
    pygame.draw.rect(screen, color,
                     pygame.Rect(x*FRAME_SIZE+1, y*FRAME_SIZE+1,
                                 FRAME_SIZE-2, FRAME_SIZE-2))
    return


if __name__ == "__main__":
    pygame.init()
    mymap = prepare_correct_map((1, 1), 0.4)
    size = max(len(mymap), len(mymap[0]))
    screen = pygame.display.set_mode([size*FRAME_SIZE,
                                      size*FRAME_SIZE])
    draw_board(screen, mymap)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
    pygame.quit()
