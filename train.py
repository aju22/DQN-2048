from envRL import Game2048
from copy import deepcopy
import numpy as np
from model import Model2048
import random
import math
from logger import TrainEpisodeLogger2048
from sklearn.utils import shuffle


class TrainModel:

    def __init__(self, NUM_EPISODES=2_00_001, BATCH_SIZE=64, resume_training=False):
        self.NUM_EPISODES = NUM_EPISODES
        self.BATCH_SIZE = BATCH_SIZE
        self.START_LEARNING_RATE = 0.0005
        self.GAMMA = 0.9
        self.EPSILON = 0.9

        self.MEM_CAPACITY = 6000
        self.resume_training = resume_training
        self.resume_ep = 0

        self.game = Game2048()
        self.logger = TrainEpisodeLogger2048()

        if self.resume_training is True:
            self.model = Model2048(pretrained=True).model
            with open("models/logs/ep_history.csv", 'r') as file:
                content = file.readlines()
            last_ep = int(content[-2:][0].split(",")[0])
            self.resume_ep = last_ep

        else:
            self.model = Model2048(pretrained=False).model

    def startTraining(self):

        maximum = -1
        episode = -1
        max_tile = -1
        total_iters = 1

        REPLAY_MEMORY = []
        REPLAY_LABELS = []
        SCORES = []

        for ep in range(self.resume_ep, self.NUM_EPISODES):
            board = self.game.new_game()

            finish = "PLAY"
            total_score = 0
            local_iters = 1
            final_tile = -1
            reward = -1

            while finish == "PLAY":

                prev_board = deepcopy(board)

                # get the required move for this state
                state = deepcopy(board)
                state = self.game.change_values(state)
                state = np.array(state, dtype=np.float32).reshape((1, 4, 4, 16))
                control_scores = self.model(state)

                # find the move with max Q value
                control_buttons = np.flip(np.argsort(control_scores), axis=1)

                # copy the Q-values as labels
                labels = deepcopy(control_scores[0].numpy())

                # generate random number for epsilon greedy approach
                num = random.uniform(0, 1)

                # store prev max
                prev_max = np.max(prev_board)
                final_tile = max(final_tile, prev_max)

                # num is less epsilon generate random move
                if num < self.EPSILON:
                    # find legal moves
                    legal_moves = self.game.findLegalMoves(prev_board)

                    if len(legal_moves) == 0:
                        finish = "LOSE"
                        continue

                    # generate random move.
                    random_dir = random.sample(legal_moves, 1)[0]

                    # apply the move
                    temp_state = deepcopy(prev_board)
                    temp_state, score = self.game.move(random_dir, temp_state)
                    total_score += score
                    finish = self.game.checkGameStatus(temp_state)

                    # get number of merges
                    empty1 = self.game.findemptyCell(prev_board)
                    empty2 = self.game.findemptyCell(temp_state)

                    if finish == "PLAY":
                        temp_state = self.game.fillTwoOrFour(temp_state)

                    board = deepcopy(temp_state)

                    # get next max after applying the move
                    next_max = np.max(temp_state)

                    # reward score
                    labels[random_dir] = math.log(score + 1)

                    # reward math.log(next_max,2)*0.1 if next_max is higher than prev max
                    labels[random_dir] += (math.log(next_max, 2))

                    if next_max == prev_max:
                        labels[random_dir] = 0

                    # reward is also the number of merges
                    labels[random_dir] += (empty2 - empty1)

                    # get the next state max Q-value
                    temp_state = self.game.change_values(temp_state)
                    temp_state = np.array(temp_state, dtype=np.float32).reshape((1, 4, 4, 16))
                    temp_scores = self.model(temp_state)

                    max_qvalue = np.max(temp_scores)

                    # final labels add gamma*max_qvalue
                    labels[random_dir] = (labels[random_dir] + self.GAMMA * max_qvalue)
                    reward = max(reward, labels[random_dir] + self.GAMMA * max_qvalue)

                else:
                    for dir in control_buttons[0]:
                        prev_state = deepcopy(prev_board)

                        # apply the LEGAl Move with max q_value
                        temp_state, score = self.game.move(dir, prev_state)

                        # if illegal move label = 0
                        if np.array_equal(prev_board, temp_state):
                            labels[dir] = 0
                            continue

                        # get number of merges
                        empty1 = self.game.findemptyCell(prev_board)
                        empty2 = self.game.findemptyCell(temp_state)

                        temp_state = self.game.fillTwoOrFour(temp_state)
                        board = deepcopy(temp_state)
                        total_score += score

                        next_max = np.max(temp_state)

                        # reward score
                        labels[dir] = math.log(score + 1)

                        # reward
                        labels[dir] += (math.log(next_max, 2))
                        if next_max == prev_max:
                            labels[dir] = 0

                        labels[dir] += (empty2 - empty1)

                        # get next max qvalue
                        temp_state = self.game.change_values(temp_state)
                        temp_state = np.array(temp_state, dtype=np.float32).reshape((1, 4, 4, 16))
                        temp_scores = self.model(temp_state)

                        max_qvalue = np.max(temp_scores)

                        # final labels
                        labels[dir] = (labels[dir] + self.GAMMA * max_qvalue)
                        reward = max(reward, (labels[dir] + self.GAMMA * max_qvalue))
                        break

                    if np.array_equal(prev_board, board):
                        finish = 'LOSE'

                # decrease the epsilon value
                if (ep > 10000) or (self.EPSILON > 0.1 and total_iters % 2500 == 0):
                    self.EPSILON = self.EPSILON / 1.005

                # change the matrix values and store them in memory
                prev_state = deepcopy(prev_board)
                prev_state = self.game.change_values(prev_state)
                prev_state = np.array(prev_state, dtype=np.float32).reshape((1, 4, 4, 16))
                REPLAY_LABELS.append(labels)
                REPLAY_MEMORY.append(prev_state)

                if len(REPLAY_MEMORY) >= self.MEM_CAPACITY:
                    X = np.array(REPLAY_MEMORY, dtype=np.float32).reshape((len(REPLAY_MEMORY), 4, 4, 16))
                    y = np.array(REPLAY_LABELS, dtype=np.float32).reshape((len(REPLAY_LABELS), 4))

                    X, y = shuffle(X, y)

                    print('Training the Model')
                    history = self.model.fit(X, y, batch_size=self.BATCH_SIZE)
                    self.logger.log_train(ep, history)

                    REPLAY_MEMORY = []
                    REPLAY_LABELS = []

                if local_iters % 400 == 0:
                    print("Episode : {}, Score : {}, Iters : {}, Finish : {}".format(ep, total_score, local_iters, finish))

                local_iters += 1
                total_iters += 1

            SCORES.append(total_score)
            print("Episode {} finished with score: {}, MaxTile : {}, result : {}".format(ep, total_score, final_tile, finish))
            self.logger.log_episode(ep, total_score, reward, final_tile)
            print()

            if (ep + 1) % 1000 == 0:
                print("Maximum Score : {} ,Maximum Tile : {}, Episode : {}".format(maximum, max_tile, episode))
            print()

            if (ep + 1) % 2000 == 0:
                self.model.save_weights(f"models_/2048Model_{ep+1}.h5")

            maximum = max(maximum, total_score)
            episode = ep
            max_tile = max(max_tile, final_tile)


trainer = TrainModel(NUM_EPISODES=2_00_001, BATCH_SIZE=64, resume_training=True)
trainer.startTraining()


