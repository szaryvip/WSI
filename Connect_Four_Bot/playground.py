import pygame
from game import Game
from player import Player

WINDOW_SIZE = (560, 560)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def main():
    # pygame initialization and screen creation
    pygame.init()
    pygame.display.set_caption("Connect Four")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    fplayer = Player(True, RED, 'minmax', 7)
    splayer = Player(False, GREEN)
    game = Game(screen, 5, 4, fplayer, splayer)
    winner = game.play()
    if winner is None:
        print("Draw!")
    elif winner == fplayer:
        print("First player won!")
    else:
        print("Second player won!")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()
