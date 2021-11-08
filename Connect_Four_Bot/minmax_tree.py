import numpy as np
from copy import deepcopy


def heuristic(board: np.array):
    pass


def is_end(board):
    pass


def children(board, is_max):
    new_boards = []
    possible_moves = []
    for index, move in enumerate(board[0]):
        if move == 0.0:
            possible_moves.append(index)
    for move in possible_moves:
        for row in range(len(board)-1, -1, -1):
            if board[row][move] == 0.0:
                board_copy = deepcopy(board)
                board_copy[row][move] = 1 if is_max else -1
                new_boards.append(board_copy)
                break
    return new_boards


def minmaxalg(board, depth, alpha, beta, is_max):
    if depth == 0 or is_end(board):
        return heuristic(board)
    if is_max:
        max_value = -10000
        for child in children(board, is_max):
            new_value = minmaxalg(board, depth-1, alpha, beta, False)
            max_value = max(max_value, new_value)
            alpha = max(alpha, new_value)
            if beta <= alpha:
                break
        return max_value
    else:
        min_value = 10000
        for child in children(board, is_max):
            new_value = minmaxalg(board, depth-1, alpha, beta, True)
            min_value = min(min_value, new_value)
            beta = min(beta, new_value)
            if beta <= alpha:
                break
        return min_value
