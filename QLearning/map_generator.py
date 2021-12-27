import random
import numpy as np
from typing import Tuple

def generate_map(
    start_point: Tuple[int, int],
    hole_prob: float,
    width: int =8,
    height: int =8
) -> np.ndarray:
    board = np.zeros([height, width])
    board[start_point[1], start_point[0]] = 1
    can_x = [x for x in range(width) if x != start_point[0]]
    can_y = [y for y in range(height) if y != start_point[1]]
    end_x = random.choice(can_x)
    end_y = random.choice(can_y)
    board[end_y, end_x] = 2
    for y in can_y:
        for x in can_x:
            if np.random.rand() < hole_prob \
               and board[y, x] != 2:
                board[y, x] = -1
    return board


def is_correct(
    mymap: np.ndarray,
    start_point: Tuple[int, int]
) -> bool:
    return True


def prepare_correct_map(
    start_point: Tuple[int, int],
    hole_prob: float,
    width: int =8,
    height: int =8
) -> np.ndarray:
    now_map = generate_map(start_point,
                           hole_prob, width, height)
    while not is_correct(now_map, start_point):
        now_map = generate_map(start_point,
                               hole_prob, width, height)
    return now_map


if __name__ == "__main__":
    print(prepare_correct_map((1,1), 0.3))
