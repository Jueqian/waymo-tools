"""
Based on waymo-open-dataset/src/waymo_open_dataset/protos/segmentation.proto at master · waymo-research/waymo

"""

from dataclasses import dataclass
from .utils import ReversibleIntEnum


class SegmentationType(ReversibleIntEnum):
    """
    3D Semantic segmentation types as per Waymo Open Dataset specification.
    """
    TYPE_UNDEFINED = 0
    TYPE_CAR = 1
    TYPE_TRUCK = 2
    TYPE_BUS = 3
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction
    # vehicles, RV, limo, tram).
    TYPE_OTHER_VEHICLE = 4
    TYPE_MOTORCYCLIST = 5
    TYPE_BICYCLIST = 6
    TYPE_PEDESTRIAN = 7
    TYPE_SIGN = 8
    TYPE_TRAFFIC_LIGHT = 9
    # Lamp post, traffic sign pole etc.
    TYPE_POLE = 10
    # Construction cone/pole.
    TYPE_CONSTRUCTION_CONE = 11
    TYPE_BICYCLE = 12
    TYPE_MOTORCYCLE = 13
    TYPE_BUILDING = 14
    # Bushes, tree branches, tall grasses, flowers etc.
    TYPE_VEGETATION = 15
    TYPE_TREE_TRUNK = 16
    # Curb on the edge of roads. This does not include road boundaries if
    # there’s no curb.
    TYPE_CURB = 17
    # Surface a vehicle could drive on. This include the driveway connecting
    # parking lot and road over a section of sidewalk.
    TYPE_ROAD = 18
    # Marking on the road that’s specifically for defining lanes such as
    # single/double white/yellow lines.
    TYPE_LANE_MARKER = 19
    # Marking on the road other than lane markers, bumps, cateyes, railtracks
    # etc.
    TYPE_OTHER_GROUND = 20
    # Most horizontal surface that’s not drivable, e.g. grassy hill,
    # pedestrian walkway stairs etc.
    TYPE_WALKABLE = 21
    # Nicely paved walkable surface when pedestrians most likely to walk on.
    TYPE_SIDEWALK = 22


@dataclass
class Segmentation:
    """
    Wrapper for segmentation information if needed as a message structure.
    """
    # 虽然 proto 里目前只是包了一层 enum，
    # 但定义这个 dataclass 可以保持与 Label 结构的对称性。
    type: SegmentationType