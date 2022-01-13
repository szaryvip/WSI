import numpy as np
from typing import Tuple, List
import random
# random.seed(1234)


def generate_map(
    start_point: Tuple[int, int],
    hole_prob: float,
    width: int = 8,
    height: int = 8
) -> np.ndarray:
    board = np.full([height, width], -1)
    correct_end = False
    while not correct_end:
        end_x = random.choice(range(1, width-1))
        end_y = random.choice(range(1, height-1))
        if (end_y, end_x) != start_point:
            correct_end = True
    board[end_y, end_x] = 200
    for y in range(height):
        for x in range(width):
            if random.uniform(0, 1) < hole_prob \
               and board[y, x] != 200 and (x, y) != start_point:
                board[y, x] = -1000
    for y in range(height):
        board[y, 0] = -1000
        board[y, width-1] = -1000
    for x in range(width):
        board[0, x] = -1000
        board[height-1, x] = -1000
    return board


def add_next_moves(point: Tuple[int, int],
                   visited: np.ndarray) -> List:
    moves = []
    if point[0]-1 >= 0:
        if visited[point[1], point[0]-1] == 0:
            moves.append((point[0]-1, point[1]))
    if point[0]+1 < len(visited[0]):
        if visited[point[1], point[0]+1] == 0:
            moves.append((point[0]+1, point[1]))
    if point[1]-1 >= 0:
        if visited[point[1]-1, point[0]] == 0:
            moves.append((point[0], point[1]-1))
    if point[1]+1 < len(visited):
        if visited[point[1]+1, point[0]] == 0:
            moves.append((point[0], point[1]+1))
    return moves


def is_correct(
    mymap: np.ndarray,
    start_point: Tuple[int, int]
) -> bool:
    tiles_visited = np.zeros([len(mymap), len(mymap[0])])
    # tiles_visited[start_point[1], start_point[0]] = 1
    to_check = add_next_moves(start_point, tiles_visited)
    while len(to_check) != 0:
        point = to_check[0]
        if tiles_visited[point[1], point[0]] == 1:
            to_check.remove(point)
            continue
        else:
            tiles_visited[point[1], point[0]] = 1
        if mymap[point[1], point[0]] == 200:
            return True
        elif mymap[point[1], point[0]] == -1000:
            to_check.remove(point)
            continue
        else:
            check = to_check
            check.remove(point)
            to_check = add_next_moves(point, tiles_visited)
            for move in check:
                to_check.append(move)
    return False


def prepare_correct_map(
    start_point: Tuple[int, int],
    hole_prob: float,
    width: int = 8,
    height: int = 8
) -> np.ndarray:
    width += 2
    height += 2
    now_map = generate_map(start_point,
                           hole_prob, width, height)
    while not is_correct(now_map, start_point):
        now_map = generate_map(start_point,
                               hole_prob, width, height)
    return now_map


if __name__ == "__main__":
    print(prepare_correct_map((1, 1), 0.3, 8, 8))
