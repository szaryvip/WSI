import numpy as np
from copy import deepcopy


def heuristic(board: np.array) -> int:
    """
    Calculates value of given board
    points:
    1 - one tile
    10 - two in row
    100 - 3 in row or 2 in row + space + one
    1000 - 4 in row
    Args:
        board (np.array): board to calculate value
    Returns:
        int: calculated value of board
    """
    value = 0
    # calc vertical

    # calc horizontal

    # calc diagonal

    return value


def is_end(board: np.array, move: list[int, int], is_max: bool) -> bool:
    """
    Check if game has winner
    Return true or false
    Args:
        board (np.array): board to start generating
        move ([int, int]): last move
        is_max (bool): if player now is max player

    Returns:
        bool: if game end or not
    """
    if move is None:
        return False
    end = True
    for place in board[0]:
        if place == 0.0:
            end = False
    if end:
        return end
    if is_max:
        whos_move = 1
    else:
        whos_move = -1
    pattern = [whos_move]*4
    in_column = len(board)
    in_row = len(board[0])
    # check vertical
    for row in range(in_column):
        if np.array_equal(board[row:row+4, move[0]], pattern):
            return True
    # chcek horizontal
    for col in range(in_row):
        if np.array_equal(board[move[1],
                          col:col+4], pattern):
            return True
    # check diagonal
    counter = 0
    for x, y in zip(range(move[0]-3, move[0]+4), range(move[1]-3,
                                                       move[1]+4)):
        if x < 0 or y < 0:
            continue
        try:
            if board[y, x] == whos_move:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        except IndexError:
            break
    counter = 0
    for x, y in zip(range(move[0]-3, move[0]+4), range(move[1]+3,
                                                       move[1]-4, -1)):
        if x < 0 or y < 0:
            continue
        try:
            if board[y, x] == whos_move:
                counter += 1
                if counter == 4:
                    return True
            else:
                counter = 0
        except IndexError:
            break
    return False


def children(board: np.array, is_max: bool):
    """Generates all possible moves for given board

    Args:
        board (np.array): board to start generating
        is_max (bool): if player now is max player

    Returns:
        np.array, [int, int]: new board and move
    """
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
                move = [move, row]
                break
    return new_boards, move


def minmaxalg(board: np.array, last_move: list[int, int], depth: int,
              alpha: int, beta: int, is_max: bool):
    """Calculate best move and return it

    Args:
        board (np.array): board to calculate
        last_move ([int,int]): move which was made
        depth (int): depth to generate
        alpha (int): min border
        beta (int): max border
        is_max (bool): if player is max player

    Returns:
        int, [int,int]: value of board, chosen move
    """
    if depth == 0 or is_end(board, last_move, is_max):
        return heuristic(board), last_move
    moves = []
    if is_max:
        max_value = -10000
        for child in children(board, is_max):
            last_move = child[1]
            new_value, move = minmaxalg(child[0], last_move, depth-1,
                                        alpha, beta, False)
            if new_value > max_value:
                moves = []
            max_value = max(max_value, new_value)
            if max_value == new_value:
                moves.append(move)
            alpha = max(alpha, new_value)
            if beta <= alpha:
                break
        return max_value, np.random.choice(moves)
    else:
        min_value = 10000
        for child in children(board, is_max):
            last_move = child[1]
            new_value, move = minmaxalg(child[0], last_move, depth-1,
                                        alpha, beta, True)
            if new_value < min_value:
                moves = []
            min_value = min(min_value, new_value)
            if min_value == new_value:
                moves.append(move)
            beta = min(beta, new_value)
            if beta <= alpha:
                break
        return min_value, np.random.choice(moves)
