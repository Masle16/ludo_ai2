import random
import time
import numpy as np
import sys

from tqdm import tqdm
from pyludo import LudoGame, LudoState
from pyludo.StandardLudoPlayers import LudoPlayerRandom
from LudoPlayerQLearningAction import LudoPlayerQLearningAction



def normal_play():
    players = []
    players = [LudoPlayerRandom() for _ in range(3)]

    epsilon = 0.05 #
    discount_factor =  0.5 #
    learning_rate = 0.25 # 
    parameters = [epsilon, discount_factor, learning_rate]

    t1 = LudoPlayerQLearningAction(parameters, chosenPolicy="greedy", QtableName='QTable', RewardName='Reward')
    players.append(t1)
    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen

    score = [0, 0, 0, 0]    

    n = 1000
    start_time = time.time()
    tqdm_1 = tqdm(range(n), ascii=True)
    for i in tqdm_1:
        tqdm_1.set_description_str(f"win rates {np.around(score/np.sum(score),decimals=2)*100}") 
        random.shuffle(players)
        ludoGame = LudoGame(players)

        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1

        for player in players: # Saving reward for QLearning player
            if type(player)==LudoPlayerQLearningAction:
                player.append_reward()

    for player in players:
        if type(player)==LudoPlayerQLearningAction:
            player.saveQTable() 
            player.saveReward()

    duration = time.time() - start_time

    print('win distribution percentage', (score/np.sum(score))*100)
    print('win distribution:', score)

def main():
   #param_optimization()
   normal_play()


if __name__ == "__main__":
    main()