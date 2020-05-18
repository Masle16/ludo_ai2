# pyludo
A python3 ludo simulator. Forked from haukuri.
### install

```
$ git clone https://github.com/RasmusHaugaard/pyludo.git
```
```
$ cd pyludo
```
```
$ pip3 install -e .
```

### examples
Visualize a game with random players
```python
from pyludo import LudoGame, LudoVisualizerStep
import numpy as np
import pyglet
import random

class LudoPlayerRandom:
    def play(self, state, dice_roll, next_states):
        """
        :param state:
            current state relative to this player
        :param dice_roll:
            [1, 6]
        :param next_states:
            np array of length 4 with each entry being the next state moving the corresponding token.
            False indicates an invalid move. 'play' won't be called, if there are no valid moves.
        :return:
            index of the token that is wished to be moved. If it is invalid, the first valid token will be chosen.
        """
        return random.choice(np.argwhere(next_states != False))

players = [LudoPlayerRandom() for _ in range(4)]
game = LudoGame(players, info=True)
window = LudoVisualizerStep(game)
print('use left and right arrow to progress game')
pyglet.app.run()
```
The equivalent of the above code can be run with the following command:
```
$ python3 -m pyludo.examples.visualizeRandomPlayerMatch
```

See pyludo/examples/randomPlayerWinStats.py for a headless example.

### state representation
The state is a numpy array of shape (4 players, 4 tokens)

`state[i]` will then be the i'th player's tokens, and `state[i][k]` will be the value of the k'th token of player i.

Home is -1, and goal is 99 for all players.
The common area is from 0 to 51 but relative to player 0.
The end lane is 52 to 56 for player 0, 57 to 61 for player 1, etc.

A LudoPlayer is always fed a relative state, where the player itself is player 0.

Note that when you move from home into the common area, you enter at position 1, not 0.

### game rules
* Always four players.
* A player must roll a 6 to enter the board.
* Rolling a 6 does not grant a new dice roll.
* Globe positions are safe positions.
* The start position outside each home is considered a globe position
* A player token landing on a single opponent token sends the opponent token home if it is not on a globe position. If the opponent token is on a globe position the player token itself is sent home.
* A player token landing on two or more opponent tokens sends the player token itself home.
* If a player token lands on one or more opponent tokens when entering the board, all opponent tokens are sent home.
* A player landing on a star is moved to the next star or directly to goal if landing on the last star.
* A player in goal cannot be moved.

