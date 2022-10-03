import random
import time

import numpy as np
from copy import deepcopy
from envRL import Game2048

game = Game2048(theme='light')


def playGame(theme, difficulty, model):
    """
    Main game loop function.
    """

    board = game.new_game()
    time.sleep(0.2)
    status = "PLAY"

    # main game loop
    while status == "PLAY":

        state = deepcopy(board)
        state = game.change_values(state)
        state = np.array(state, dtype=np.float32).reshape((1, 4, 4, 16))

        control_scores = model(state)
        control_buttons = np.flip(np.argsort(control_scores), axis=1)

        key = control_buttons[0][0]
        legal_moves = game.findLegalMoves(board)

        while key not in legal_moves:
            key = random.randint(0, 3)

        if random.random() < 0.2:
            key = random.randint(0, 3)

        print(f"applying move: {key}")
        new_board, score = game.move(key, board)
        print(new_board)

        if new_board != board:

            board = game.fillTwoOrFour(new_board)
            game.display(board, theme)
            # update game status
            status = game.checkGameStatus(board, difficulty)
            # check if the game is over
            board, status = game.winCheck(board, status)

        else:
            board = new_board
