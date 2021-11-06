"""
- kiedy budujemy drzewo minimax o głębokości D to warto sprawdzić różne
    wartości tego D<=D_Max
- należałoby sprawdzić, czy gra nie faworyzuje, któregoś z graczy
     i wykazać w raporcie wygraną obu stron.
     Jeżeli istnieje możliwość remisu to też warto byłoby to pokazać.
- gry odbywają się w sposób losowy, dlatego też na potrzebę pomiarów
     należałoby wykonać N prób
- wybór następnego ruchu przez algorytm dla zbioru tak samo ocenionych
     stanów powinien wykonywać się w sposób losowy """

import pygame
from game import Game
from player import Player

WINDOW_SIZE = (560, 560)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def main():
    pygame.init()
    pygame.display.set_caption("Connect Four")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    fplayer = Player(True, RED)
    splayer = Player(False, GREEN)
    game = Game(screen, 5, 4, fplayer, splayer)
    game.play()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()
