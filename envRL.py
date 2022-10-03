import random
import numpy as np
import math
import json
import pygame
from pygame.locals import *
from copy import deepcopy
import time
import sys


class Game2048:

    def __init__(self, theme='light'):
        self.c = json.load(open("constants.json", "r"))
        pygame.init()
        self.screen = pygame.display.set_mode((self.c["size"], self.c["size"]))
        self.my_font = pygame.font.SysFont(self.c["font"], self.c["font_size"], bold=True)
        WHITE = (255, 255, 255)
        self.theme = theme
        self.screen.fill(tuple(self.c["colour"][theme]["background"]))
        self.box = self.c["size"] // 4
        self.padding = self.c["padding"]

    def new_game(self):

        self.screen.blit(self.my_font.render("NEW GAME!", 1, (0, 0, 0)), (130, 225))
        pygame.display.update()
        # wait for 1 second before starting over
        time.sleep(0.5)
        self.screen.fill(tuple(self.c["colour"][self.theme]["background"]))
        pygame.display.update()
        time.sleep(0.5)
        board = [[0] * 4 for _ in range(4)]
        board = self.fillTwoOrFour(board, iter=2)
        return board

    def move(self, direction, board):
        """
        Call functions to move & merge in the specified direction.

        Returns:
            (list): updated board after move completion
        """
        if direction == 0:
            return self.moveUp(board)
        if direction == 1:
            return self.moveDown(board)
        if direction == 2:
            return self.moveLeft(board)
        if direction == 3:
            return self.moveRight(board)

    def checkGameStatus(self, board, max_tile=2048):
        """
        Update the game status by checking if the max. tile has been obtained.
        Returns:
            (str): game status WIN/LOSE/PLAY
        """
        flat_board = [cell for row in board for cell in row]
        if max_tile in flat_board:
            # game has been won if max_tile value is found
            return "WIN"

        for i in range(4):
            for j in range(4):
                # check if a merge is possible
                if j != 3 and board[i][j] == board[i][j + 1] or \
                        i != 3 and board[i][j] == board[i + 1][j]:
                    return "PLAY"

        if 0 not in flat_board:
            return "LOSE"
        else:
            return "PLAY"

    def checkSame(self, board, new_board):

        for i in range(4):
            for j in range(4):
                if not board[i][j] == new_board[i][j]:
                    return False

        return True

    def fillTwoOrFour(self, board, iter=1):
        """
        Randomly fill 2 or 4 in available spaces on the board.
        Returns:
            board (list): updated game board
        """
        for _ in range(iter):
            a = random.randint(0, 3)
            b = random.randint(0, 3)
            while board[a][b] != 0:
                a = random.randint(0, 3)
                b = random.randint(0, 3)

            if sum([cell for row in board for cell in row]) in (0, 2):
                board[a][b] = 2
            else:
                board[a][b] = random.choice((2, 4))

        return board

    def moveLeft(self, board):
        """
        Move and merge tiles to the left.

        Parameters:
            board (list): game board
        Returns:
            board (list): updated game board
        """
        # initial shift
        self.shiftLeft(board)
        score = 0

        # merge cells
        for i in range(4):
            for j in range(3):
                if board[i][j] == board[i][j + 1] and board[i][j] != 0:
                    board[i][j] *= 2
                    score += board[i][j]
                    board[i][j + 1] = 0
                    j = 0

        # final shift
        self.shiftLeft(board)
        return board, score

    def moveUp(self, board):
        """
        Move ane merge tiles upwards.

        Returns:
            board (list): updated game board
        """
        board = self.rotateLeft(board)
        board, score = self.moveLeft(board)
        board = self.rotateRight(board)
        return board, score

    def moveRight(self, board):
        """
        Move and merge tiles to the right.

        Returns:
            board (list): updated game board
        """
        # initial shift
        self.shiftRight(board)
        score = 0

        # merge cells
        for i in range(4):
            for j in range(3, 0, -1):
                if board[i][j] == board[i][j - 1] and board[i][j] != 0:
                    board[i][j] *= 2
                    score += board[i][j]
                    board[i][j - 1] = 0
                    j = 0

        # final shift
        self.shiftRight(board)
        return board, score

    def moveDown(self, board):
        """
        Move and merge tiles downwards.
        Returns:
            board (list): updated game board
        """
        board = self.rotateLeft(board)
        board, score = self.moveLeft(board)
        self.shiftRight(board)
        board = self.rotateRight(board)
        return board, score

    def shiftLeft(self, board):
        """
        Perform tile shift to the left.
        """
        # remove 0's in between numbers
        for i in range(4):
            nums, count = [], 0
            for j in range(4):
                if board[i][j] != 0:
                    nums.append(board[i][j])
                    count += 1
            board[i] = nums
            board[i].extend([0] * (4 - count))

    def shiftRight(self, board):
        """
        Perform tile shift to the right.
        """
        # remove 0's in between numbers
        for i in range(4):
            nums, count = [], 0
            for j in range(4):
                if board[i][j] != 0:
                    nums.append(board[i][j])
                    count += 1
            board[i] = [0] * (4 - count)
            board[i].extend(nums)

    def rotateLeft(self, board):
        """
        90 degree counter-clockwise rotation.
        Returns:
            b (list): new game board after rotation
        """
        b = [[board[j][i] for j in range(4)] for i in range(3, -1, -1)]
        return b

    def rotateRight(self, board):
        """
        270 degree counter-clockwise rotation.
        Returns:
            (list): new game board after rotation
        """
        b = self.rotateLeft(board)
        b = self.rotateLeft(b)
        return self.rotateLeft(b)

    def change_values(self, X):
        power_mat = np.zeros(shape=(1, 4, 4, 16), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if X[i][j] == 0:
                    power_mat[0][i][j][0] = 1.0
                else:
                    power = int(math.log(X[i][j], 2))
                    power_mat[0][i][j][power] = 1.0
        return power_mat

    def findemptyCell(self, mat):
        count = 0
        for i in range(len(mat)):
            for j in range(len(mat)):
                if mat[i][j] == 0:
                    count += 1
        return count

    def findLegalMoves(self, board):

        controls = {"up": 0, "down": 1, "left": 2, "right": 3}
        legal_moves = []

        for key, val in controls.items():
            temp_board = deepcopy(board)
            dir = controls[key]
            temp_board, score = self.move(dir, temp_board)
            if np.array_equal(temp_board, board):
                continue
            else:
                legal_moves.append(val)

        return legal_moves

    def winCheck(self, board, status):

        if status != "PLAY":
            size = self.c["size"]
            # Fill the window with a transparent background
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill(self.c["colour"][self.theme]["over"])
            self.screen.blit(s, (0, 0))

            # Display win/lose status
            if status == "WIN":
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER!"

            self.screen.blit(self.my_font.render(msg, 1, (0, 0, 0)), (140, 180))
            # Ask user to play again
            self.screen.blit(self.my_font.render(
                "Play again? (y/ n)", 1, (0, 0, 0)), (80, 255))

            pygame.display.update()

            while True:
                for event in pygame.event.get():
                    if event.type == QUIT or \
                            (event.type == pygame.KEYDOWN and event.key == K_n):
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.KEYDOWN and event.key == K_y:
                        # 'y' is pressed to start a new game
                        board = self.new_game()
                        return board, "PLAY"

        return board, status

    def display(self, board, theme):

        for i in range(4):
            for j in range(4):
                colour = tuple(self.c["colour"][self.theme][str(board[i][j])])
                pygame.draw.rect(self.screen, colour, (j * self.box + self.padding,
                                                       i * self.box + self.padding,
                                                       self.box - 2 * self.padding,
                                                       self.box - 2 * self.padding), 0)
                if board[i][j] != 0:
                    if board[i][j] in (2, 4):
                        text_colour = tuple(self.c["colour"][self.theme]["dark"])
                    else:
                        text_colour = tuple(self.c["colour"][self.theme]["light"])
                    # display the number at the centre of the tile
                    self.screen.blit(self.my_font.render("{:>4}".format(
                        board[i][j]), 1, text_colour),
                        # 2.5 and 7 were obtained by trial and error
                        (j * self.box + 2.5 * self.padding, i * self.box + 7 * self.padding))

        time.sleep(0.3)
        pygame.display.update()
