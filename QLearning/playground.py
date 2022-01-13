from game import Game
import matplotlib.pyplot as plt
import pygame
import numpy as np
np.random.seed(1234)


def testing(start_point, max_iter, discount,
            learn_rate, epsilon, train_number, show):
    if show:
        pygame.init()
    moves = 0
    game = Game(start_point, 0.3, max_iter, discount,
                learn_rate, epsilon, 8, 8, show)
    for _ in range(train_number):
        correct_start = False
        while not correct_start:
            start_x = np.random.randint(1, 9)
            start_y = np.random.randint(1, 9)
            if game._my_map[start_y, start_x] == -1:
                correct_start = True
        game.set_start_point((start_x, start_y))
        moves += game.play()
    if show:
        game.set_start_point((1, 1))
        game.show_game()
        pygame.quit()
    return moves


if __name__ == "__main__":
    # np.random.seed(1234)
    results = []
    x = []
    for disc in range(1, 11, 1):
        disc /= 10
        print(disc)
        result = 0
        x.append(disc)
        to_avg = []
        for _ in range(25):
            res = testing((1, 1), 1000, 0.9, 0.8, disc, 100, False)
            to_avg.append(res)
        result = np.average(to_avg)
        results.append(result)

    plt.plot(x, results)
    plt.xlabel("epsilon")
    plt.ylabel("number of moves during training")
    plt.show()
