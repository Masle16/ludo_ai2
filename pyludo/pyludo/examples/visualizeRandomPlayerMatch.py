from pyludo import LudoGame, LudoPlayerRandom, LudoVisualizerStep
import pyglet

game = LudoGame([LudoPlayerRandom() for _ in range(4)], info=True)
window = LudoVisualizerStep(game)
print('use left and right arrow to progress game')
pyglet.app.run()
