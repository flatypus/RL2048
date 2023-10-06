import os
import tensorflow as tf
from two_oh_four_eight import Game

current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "model_output/2048_3240.h5")
model = tf.keras.models.load_model(path)
model.load_weights(path + "_weights.h5")
tf.keras.utils.disable_interactive_logging()

game = Game()

while True:
    state = game._conform_to_output()
    action = model.predict(state)
    action = tf.argmax(action, axis=1)
    action = action.numpy()[0]
    game.step(action)
    game.render()
    if game.done:
        break
