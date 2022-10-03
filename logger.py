import csv
from os.path import exists


class TrainEpisodeLogger2048:
    """
    TrainEpisodeLogger2048 is used to show the
    NN training results (max tile reached, average reward, etc.) and to save the results in a CSV file.
    """

    def __init__(self):

        self.train_idx = 0

        self.filePath_train = 'models/logs/train_history.csv'
        self.filePath_eps = 'models/logs/ep_history.csv'

        self.csv_writer_ep = self.getCSVWriter(self.filePath_eps,
                                               headers=['episode', 'score', 'max_reward', 'max_tile'])
        self.csv_writer_train = self.getCSVWriter(self.filePath_train,
                                                  headers=['idx', 'episode', 'loss', 'mean_abs_error'])

    def getCSVWriter(self, filePath, headers):

        if exists(filePath):
            csv_file = open(filePath, "a")  # a = append
            csv_writer = csv.writer(csv_file, delimiter=',')
        else:
            csv_file = open(filePath, "w")  # w = write (clear and restart)
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(headers)

        return csv_writer

    def log_episode(self, episode, score, max_reward, max_tile):

        self.csv_writer_ep.writerow((episode + 1, score, max_reward, max_tile))

    def log_train(self, episode, history):

        self.train_idx += 1
        loss = history.history['loss'][0]
        mean_abs_err = history.history['mean_absolute_error'][0]

        self.csv_writer_train.writerow((self.train_idx, episode + 1, loss, mean_abs_err))
