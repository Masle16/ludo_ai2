from pyludo import LudoGame, LudoVisualizerStep
from pyludo.StandardLudoPlayers import LudoPlayerRandom, LudoPlayerFast, LudoPlayerAggressive, LudoPlayerDefensive
import pyglet


players = [
    LudoPlayerRandom(),
    LudoPlayerFast(),
    LudoPlayerAggressive(),
    LudoPlayerDefensive(),
]

game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)
print('use left and right arrow to progress game')
pyglet.app.run()
