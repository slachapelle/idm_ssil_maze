from dataclasses import dataclass


@dataclass
class DeepMindColor:
    obstacle = (5, 55, 66)
    free = (232, 240, 242)
    # agent = (204, 85, 0)
    agent = (240, 84, 84)
    goal = (51, 255, 51)
    button = (102, 0, 204)
    interruption = (255, 0, 255)
    box = (0, 102, 102)
    lava = (255, 0, 0)
    water = (0, 0, 255)
