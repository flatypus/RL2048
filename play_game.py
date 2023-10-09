from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import os
import time
import tensorflow as tf


current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "model_output")
model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
model.load_weights(os.path.join(path, "weights.h5"))
tf.keras.utils.disable_interactive_logging()


options = Options()
options.add_argument("--verbose")
# options.add_argument("--headless")

driver = webdriver.Chrome(options=options)
driver.get("https://play2048.co/")


def get_state():
    tiles = driver.find_elements(By.CLASS_NAME, "tile")
    # print all the class names of the elements
    grid = [[None for _ in range(4)] for _ in range(4)]
    for tile in tiles:
        class_names = tile.get_attribute("class").split()
        row = int(class_names[2].split("-")[2]) - 1
        col = int(class_names[2].split("-")[3]) - 1
        points = int(class_names[1].split("-")[1])
        grid[col][row] = points

    max_exponent = 13
    grid = np.reshape(grid, -1)
    grid = [x if x else 0 for x in grid]
    grid = np.array([encode(x, max_exponent) for x in grid])
    grid = np.reshape(grid, (4, 4, -1))
    return grid


def act(action):
    value = ""
    # action: 0, 1, 2, 3
    # move:   up, right, down, left (use arrow keys)
    match action:
        case 0:
            value = "\uE013"
        case 1:
            value = "\uE014"
        case 2:
            value = "\uE015"
        case 3:
            value = "\uE012"

    actions = ActionChains(driver)
    actions.send_keys(value)
    actions.perform()


def encode(x, max_exponent):
    return [1 if x == 2 **
            i else 0 for i in range(max_exponent)]


def loop_actions(prev_state, actions):
    for i in range(len(actions)):
        action = actions[i][0]
        act(action)
        state = get_state()
        if not np.array_equal(prev_state, state):
            return


time.sleep(3)


while True:
    try:
        state = get_state()
        options = model.predict(state)
        actions = sorted(
            enumerate(options[0]), key=lambda x: x[1], reverse=True)
        prev_state = state
        loop_actions(prev_state, actions)
    except:
        pass
    # time.sleep(0.02)
