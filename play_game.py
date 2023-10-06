from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import numpy as np
import os
import time
import tensorflow as tf


current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, "model_output/2048_3240.h5")
weight_path = os.path.join(
    current_dir, "model_output/2048_3240.h5_weights.h5")
model = tf.keras.models.load_model(path)
model.load_weights(weight_path)
tf.keras.utils.disable_interactive_logging()


options = Options()
options.add_argument("--verbose")
# options.add_argument("--headless")

driver = webdriver.Chrome(options=options)
driver.get("https://play2048.co/")

time.sleep(3)
while True:
    try:
        tiles = driver.find_elements(By.CLASS_NAME, "tile")
        # print all the class names of the elements
        grid = [
            [[0], [0], [0], [0]],
            [[0], [0], [0], [0]],
            [[0], [0], [0], [0]],
            [[0], [0], [0], [0]]
        ]
        for tile in tiles:
            class_names = tile.get_attribute("class").split()
            row = int(class_names[2].split("-")[2]) - 1
            col = int(class_names[2].split("-")[3]) - 1
            points = int(class_names[1].split("-")[1])
            grid[col][row] = [points]
        state = np.array([grid])
        options = model.predict(state)
        action = np.argmax(options, axis=1)
        print(action)
        value = ""
        # action: 0, 1, 2, 3
        # move:   up, right, down, left (use arrow keys)
        match action[0]:
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
    except:
        pass
    time.sleep(0.02)
