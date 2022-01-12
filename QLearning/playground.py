from game import Game
import matplotlib.pyplot as plt
import pygame
import numpy as np
np.random.seed(1234)


def testing(start_point, max_iter, discount,
            learn_rate, epsilon, train_number):
    pygame.init()
    game = Game(start_point, 0.3, max_iter, discount, learn_rate, epsilon)
    for _ in range(train_number):
        correct_start = False
        while not correct_start:
            start_x = np.random.randint(1, 9)
            start_y = np.random.randint(1, 9)
            if game._my_map[start_y, start_x] == -1:
                correct_start = True
        game.set_start_point((start_x, start_y))
        game.play()
    game.set_start_point((1, 1))
    game.show_game()
    pygame.quit()


if __name__ == "__main__":
    np.random.seed(1234)
    testing((1, 1), 100, 1, 1, 1, 100)
