from enum import Enum

class DECISION_STATE(Enum):
    STOP = 0
    ADAPT_TO_FORWARD_SPEED = 1
    OVERTAKE = 2
    START_NAVIGATION = 3
    STOP_NAVIGATION = 4