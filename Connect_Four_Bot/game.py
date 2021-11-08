import pygame
import numpy as np
from time import sleep
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
        """
        Constructor of Game object
        Parameters:
        - screen [pygame.Surface] - window where game will be created
        - in_row [int] - how many tiles in row
        - in_column [int] - how many tiles in column
        - (f/s)player [Player] - players
                 """
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
        """
        Initialize board for game and draw it"""
        for col in range(self._in_column):
            for row in range(self._in_row):
                pygame.draw.circle(self._screen, (106, 0, 206),
                                   [(2*row+1)*self._rad,
                                    self._size[1]-(2*col+1)*self._rad],
                                   self._rad, 2)
        pygame.display.flip()

    def draw_move(self, player) -> tuple[int, int]:
        """
        Draw player's move
        Parameter:
        -player [Player] - who is moving now"""
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
        return (col_move, self._in_column-row_move-1)

    def is_winner(self, move, player) -> bool:
        """
        Check if game has winner
        Return true or false"""
        whos_move = 1 if player.is_max() else -1
        pattern = [whos_move]*4
        # check vertical
        for row in range(self._in_column-3):
            if np.array_equal(self._board[row:row+4, move[0]], pattern):
                return True
        # chcek horizontal
        for col in range(self._in_row-3):
            if np.array_equal(self._board[move[1],
                              col:col+4], pattern):
                return True
        # check diagonal
        counter = 0
        for x, y in zip(range(move[0]-3, move[0]+4), range(move[1]-3,
                                                           move[1]+4)):
            if x < 0 or y < 0:
                continue
            try:
                if self._board[y, x] == whos_move:
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
                if self._board[y, x] == whos_move:
                    counter += 1
                    if counter == 4:
                        return True
                else:
                    counter = 0
            except IndexError:
                break
        return False

    def is_draw(self) -> bool:
        """
        Check if game ended with draw"""
        end = True
        for place in self._board[0]:
            if place == 0.0:
                end = False
        return end

    def play(self) -> Player:
        """
        Main function to run game between bots"""
        self.draw_board()
        while not self.is_draw():
            move = self.draw_move(self._fplayer)
            if self.is_winner(move, self._fplayer):
                return self._fplayer
            elif self.is_draw():  # must have if board have odd number of tiles
                break
            sleep(0.4)
            move = self.draw_move(self._splayer)
            if self.is_winner(move, self._splayer):
                return self._splayer
            sleep(0.4)
        return None
