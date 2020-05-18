from pyludo import LudoGame
from pyludo.StandardLudoPlayers import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
import random
import time

players = [
    LudoPlayerRandom(),
    LudoPlayerFast(),
    LudoPlayerAggressive(),
    LudoPlayerDefensive(),
]

scores = {}
for player in players:
    scores[player.name] = 0

n = 1000

start_time = time.time()
for i in range(n):
    random.shuffle(players)
    ludoGame = LudoGame(players)
    winner = ludoGame.play_full_game()
    scores[players[winner].name] += 1
    print('Game ', i, ' done')
duration = time.time() - start_time

print('win distribution:', scores)
print('games per second:', n / duration)
