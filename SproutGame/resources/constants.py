from enum import Enum


MAXIMUM_FORCE_MAGNITUDE = 0.01

DEFAULT_OPTIMUM_LENGTH = 0.1

CANVAS_SIZE = 640

VERTEX_SIZE = 10
EDGE_WIDTH = 2
PATH_WIDTH = 4


LOADING_ANIMATION_PATH = "SproutGame/resources/thinking.gif"

class Color(Enum):
    RED = 'red'
    GREEN = 'green'
    BLUE = 'blue'
    YELLOW = 'yellow'
    GREY = 'grey'


class LineStyle(Enum):
    SOLID = 'solid'
    DASHED = (16, 8)
    DASHED_DEMO = 'dashed'

