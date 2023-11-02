from src.yolo_utils import get_anchors, read_class_names
import numpy as np

INPUT_SIZE = 416
ANCHORS = "src/model_dependencies/yolov4_anchors.txt"
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]
ANCHORS = get_anchors(ANCHORS)
STRIDES = np.array(STRIDES)
CLASS_NAMES = read_class_names("src/model_dependencies/coco.names")