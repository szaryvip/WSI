import pygame
import numpy as np
from player import Player


class Game:
    _in_row = 5
    _in_column = 4
    _screen = None
    _fplayer = None
    _splayer = None
    _board = None
    _size = []
    _rad = 0

    def __init__(self, screen: pygame.Surface, in_row: int, in_column: int,
                 fplayer: Player, splayer: Player):
        if in_row >= in_column:
            self._in_row = in_row
            self._in_column = in_column
        elif in_row < 5 or in_column < 4:
            print("Board to small. Program will generate minimal board.\n")
        else:
            print("""Board have to contain more or equal tiles in row
        than in column. Program will generate board with the same values
        in row and column.""")
            self._in_row = in_row
            self._in_column = in_row
        self._screen = screen
        self._fplayer = fplayer
        self._splayer = splayer
        self._board = np.zeros((self._in_column, self._in_row))
        self._size = self._screen.get_size()
        self._rad = min(self._size)/(2*self._in_row)

    def draw_board(self) -> None:
        for col in range(self._in_column):
            for row in range(self._in_row):
                pygame.draw.circle(self._screen, (106, 0, 206),
                                   [(2*row+1)*self._rad,
                                    self._size[1]-(2*col+1)*self._rad],
                                   self._rad, 2)
        pygame.display.flip()

    def draw_move(self, player) -> None:
        col_move = player.make_move(self._board)
        color = player.get_color()
        row_move = 0
        for row in self._board[::-1]:
            if row[col_move] == 0.0:
                break
            row_move += 1
        pygame.draw.circle(self._screen, color,
                           [self._rad*(2*col_move+1),
                            self._size[1]-self._rad*(2*row_move+1)],
                           self._rad-2)
        value = 1 if player == self._fplayer else -1
        self._board[self._in_column-row_move-1][col_move] = value
        pygame.display.flip()

    def is_winner(self) -> Player:
        pass

    def is_end(self) -> bool:
        end = True
        for place in self._board[0]:
            if place == 0.0:
                end = False
        return end

    def play(self) -> None:
        self.draw_board()
        while not self.is_end():
            self.draw_move(self._fplayer)
            self.draw_move(self._splayer)
