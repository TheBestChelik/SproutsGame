from enum import Enum


MAXIMUM_FORCE_MAGNITUDE = 0.01

DEFAULT_OPTIMUM_LENGTH = 0.1

CANVAS_SIZE = 640

VERTEX_SIZE = 10
EDGE_WIDTH = 2
PATH_WIDTH = 4
# SPOT_SIZE = 15


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


class ErrorMessage(Enum):
    MOVE_START_FROM_VERTEX = "Move has to start from Spot, not Vertex"
    SPOT_IS_DEAD = "This spot has not enought liberties"
    PATH_INTERSECTION = "You can not intersect the path"
    ILLIGAL_MOVE = "You can move only along the dashed edges"
    NO_MOVE_TO_CANCELL = "There is nothing to cancel"
    CANCEL_NOT_LAST_VERTEX = "You can cancel only the last vertex"
