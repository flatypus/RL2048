import os
import tensorflow as tf
import numpy as np
from two_oh_four_eight import Game

current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "model_output")
model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
model.load_weights(os.path.join(path, "weights.h5"))
tf.keras.utils.disable_interactive_logging()

game = Game()

while True:
    state = game.get_state()
    action = model.predict(state)
    actions = sorted(enumerate(action[0]), key=lambda x: x[1], reverse=True)
    prev_state = game.grid
    for i in range(len(actions)):
        action = actions[i][0]
        game.step(action)
        if not np.array_equal(prev_state, game.grid):
            break

    game.render()
    if game.done:
        break
