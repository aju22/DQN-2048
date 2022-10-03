from game import playGame
from model import Model2048


if __name__ == "__main__":

    #load model
    model = Model2048(pretrained=True).model

    # display the start screen 
    playGame(theme='light', difficulty=2048, model=model)
