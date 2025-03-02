from enum import Enum

class AttackTypes(Enum):
    GuassianAttack = "guassian-attack"
    LabelFlipping = "label-flipping"
    BackDoor = "back-door"
    MultiLabelFlipping = "multi-label-flipping"
