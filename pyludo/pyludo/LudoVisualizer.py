import os
import pyglet
from pyglet.window import key
from pyludo.LudoGame import LudoState
from pyludo.utils import player_colors

home_sprite_positions = [
    [(1, 10), (4, 10), (4, 13), (1, 13)],
    [(10, 10), (13, 10), (13, 13), (10, 13)],
    [(10, 1), (13, 1), (13, 4), (10, 4)],
    [(1, 1), (4, 1), (4, 4), (1, 4)],
]

pos_to_index = [
    (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
    (7, 14), (8, 14), (8, 13), (8, 12), (8, 11), (8, 10), (8, 9), (9, 8), (10, 8), (11, 8), (12, 8), (13, 8),
    (14, 8), (14, 7), (14, 6), (13, 6), (12, 6), (11, 6), (10, 6), (9, 6), (8, 5), (8, 4), (8, 3), (8, 2),
    (8, 1), (8, 0), (7, 0), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (5, 6), (4, 6), (3, 6), (2, 6),
    (1, 6), (0, 6), (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (7, 13), (7, 12), (7, 11), (7, 10), (7, 9),
    (13, 7), (12, 7), (11, 7), (10, 7), (9, 7), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)
]


class LudoVisualizer(pyglet.window.Window):
    def __init__(self, state=None):
        super(LudoVisualizer, self).__init__()

        self.state = LudoState() if state is None else state
        # set scaling
        self.width, self.height = 600, 600
        bg_img = self.load_img('ludoboard.png')
        self.scaling = min(bg_img.height, self.height) / max(bg_img.height, self.height)
        # load sprites
        self.backgroundSprite = self.load_sprite('ludoboard.png')
        self.playerSprites = [self.load_sprite(color + 'Player.png') for color in player_colors]
        self.globeSprite = self.load_sprite('globe.png')
        self.starSprite = self.load_sprite('star.png')

        # force draw first draw
        self.switch_to()
        self.dispatch_event("on_draw")
        self.dispatch_events()
        self.flip()

    def index_to_pixels(self, index, token_id=-1, offset_scale=2):
        offset = [(0, 0), (-1, -1), (1, -1), (1, 1), (-1, 1)][token_id + 1]
        off_x = offset[0] * offset_scale
        off_y = offset[1] * offset_scale
        return index[0] * self.scaling * 50.0 + off_x, index[1] * self.scaling * 50.0 + off_y

    def on_draw(self):
        self.clear()
        self.backgroundSprite.draw()
        for i in range(4):
            offset = i * 13
            for j in [1, 9]:
                self.globeSprite.position = self.index_to_pixels(pos_to_index[offset + j])
                self.globeSprite.draw()
            for j in [6, 12]:
                self.starSprite.position = self.index_to_pixels(pos_to_index[offset + j])
                self.starSprite.draw()
        for player_id, player_tokens in enumerate(self.state):
            sprite = self.playerSprites[player_id]
            for token_id, token_pos in enumerate(player_tokens):
                if token_pos == -1:
                    sprite.position = self.index_to_pixels(home_sprite_positions[player_id][token_id])
                elif token_pos != 99:
                    sprite.position = self.index_to_pixels(pos_to_index[token_pos], token_id=token_id)
                elif token_pos == 99:
                    end_pos = [(6, 7), (7, 8), (8, 7), (7, 6)][player_id]
                    sprite.position = self.index_to_pixels(end_pos, token_id=token_id, offset_scale=5)
                sprite.draw()

    @staticmethod
    def load_img(local_path):
        assets_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        return pyglet.image.load(os.path.join(assets_folder, local_path))

    def load_sprite(self, local_path):
        img = self.load_img(local_path)
        sprite = pyglet.sprite.Sprite(img)
        sprite.scale = self.scaling
        return sprite


class LudoVisualizerStep(LudoVisualizer):
    def __init__(self, game):
        super(LudoVisualizerStep, self).__init__()
        self.game = game
        self.states = [game.state]
        self.state_index = 0

    def on_key_press(self, symbol, _):
        if symbol == key.LEFT:
            self.state_index = max(0, self.state_index - 1)
        if symbol == key.RIGHT:
            self.state_index += 1
            while self.state_index >= len(self.states):
                self.game.step()
                self.states.append(self.game.state)
        self.state = self.states[self.state_index]
