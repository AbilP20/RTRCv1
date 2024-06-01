from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import random
from typing import Tuple
from paths import *
from logger import *

lg = logging.getLogger(__name__)

def load_models() -> Tuple[YOLO, YOLO, YOLO, dict]:
    """
    Returns the 3 YOLO models(detection, segmentation and classification) and 
    a modified class names dictionary of detection model
    """
    try:
        model_bbox = YOLO(os.path.join(MODEL_DIR, "det.pt")) # detection model
        bbox_class_names = model_bbox.names
        bbox_class_names[17] = '20'
        model_mask = YOLO(os.path.join(MODEL_DIR, "seg.pt")) # segmentation model
        model_classify = YOLO(os.path.join(MODEL_DIR, "cls.pt")) # classification model
        lg.info("load_models() - Models loaded.")
        return model_bbox, model_mask, model_classify, bbox_class_names
    except FileNotFoundError:
        lg.warning("load_models() - YOLO Model file not found.")
    except Exception as e:
        lg.warning(f"load_models() - {e}")
        
def load_hud_icons() -> Tuple[np.ndarray]:
    """
    Returns a tuple containing all the hud icons as numpy.ndarray type
    """
    try:
        png_20 = cv2.imread(PNG_20, cv2.IMREAD_UNCHANGED)
        png_30 = cv2.imread(PNG_30, cv2.IMREAD_UNCHANGED)
        png_40 = cv2.imread(PNG_40, cv2.IMREAD_UNCHANGED)
        png_50 = cv2.imread(PNG_50, cv2.IMREAD_UNCHANGED)
        png_60 = cv2.imread(PNG_60, cv2.IMREAD_UNCHANGED)
        png_70 = cv2.imread(PNG_70, cv2.IMREAD_UNCHANGED)
        png_80 = cv2.imread(PNG_80, cv2.IMREAD_UNCHANGED)
        png_100 = cv2.imread(PNG_100, cv2.IMREAD_UNCHANGED)
        png_120 = cv2.imread(PNG_120, cv2.IMREAD_UNCHANGED)
        png_stop = cv2.imread(PNG_STOP, cv2.IMREAD_UNCHANGED)
        png_crosswalk = cv2.imread(PNG_CROSSWALK, cv2.IMREAD_UNCHANGED)
        png_priority_road = cv2.imread(PNG_PRIORITY_ROAD, cv2.IMREAD_UNCHANGED)
        png_speed_hump = cv2.imread(PNG_SPEED_HUMP, cv2.IMREAD_UNCHANGED)
        png_yield = cv2.imread(PNG_YIELD, cv2.IMREAD_UNCHANGED)
        png_red_light = cv2.imread(PNG_RED_LIGHT, cv2.IMREAD_UNCHANGED)
        png_no_entry = cv2.imread(PNG_NO_ENTRY, cv2.IMREAD_UNCHANGED)
        png_car_vicinity = cv2.imread(PNG_CAR_VICINITY, cv2.IMREAD_UNCHANGED)
        png_car_closeby = cv2.imread(PNG_CAR_CLOSEBY, cv2.IMREAD_UNCHANGED)
        png_person_vicinity = cv2.imread(PNG_PERSON_VICINITY, cv2.IMREAD_UNCHANGED)
        png_person_closeby = cv2.imread(PNG_PERSON_CLOSEBY, cv2.IMREAD_UNCHANGED)
        png_road = cv2.imread(PNG_ROAD, cv2.IMREAD_UNCHANGED)
        hud_frame = cv2.imread(HUD_BG)
        lg.info("load_hud_icons() - HUD icons loaded.")
        return (png_20, png_30, png_40, png_50, png_60, png_70, png_80, png_100, png_120, 
                png_stop, png_crosswalk, png_priority_road, png_speed_hump, png_yield, 
                png_red_light, png_no_entry, png_car_vicinity, png_car_closeby, png_person_vicinity, 
                png_person_closeby, png_road, hud_frame)
    except FileNotFoundError as e:
        lg.warning("load_hud_icons() - HUD Image file not found.")
    except Exception as e:
        lg.warning(f"load_hud_icons() - {e}")

def load_vid_list(source):
    """
    Returns a list containing the paths of test videos
    """
    try:
        vids = []
        try:
            source = int(source)
            for i in os.listdir(VID_PATH):
                vids.append(os.path.join(VID_PATH,i))
            if source >= len(vids):
                raise IndexError
            else:
                vid_index = source
                lg.info("load_vid_list() - Source video list loaded.")
            return vids, vid_index
        except IndexError:
            lg.warning("load_vid_list() - Specified video index is out of range.")
        except ValueError:
            vids.append(source)
            vid_index = 0
            lg.info("load_vid_list() - Source video list loaded.")
            return vids, vid_index
    except FileNotFoundError:
        lg.warning("load_vid_list() - Source video path not found.")
    except Exception as e:
        lg.warning(f"load_vid_list() - {e}")   

def get_vid_info(video_path:str) -> Tuple[int, int, int, int] :
    """
    Returns the following attributes of any video - width, height, fps and total number of frames 
    """
    width, height, fps, total_frames = sv.VideoInfo.from_video_path(video_path).__dict__.values()
    lg.info(f"get_vid_info() - Video Info {width, height, fps, total_frames}")
    return width, height, fps, total_frames

def get_cur_speed() -> int:
    """
    Returns random speed from the defined list of speeds which acts as the current speed of the vehicle
    """
    speeds = [20, 35, 40, 50, 65, 70, 80, 95, 115]
    cur_speed = random.choice(speeds)
    return cur_speed

def get_wc_zone_coordinates(orig_width:int, orig_height:int) -> Tuple[np.ndarray, np.ndarray,
                                                                            sv.detection.tools.polygon_zone.PolygonZone,
                                                                            sv.detection.tools.polygon_zone.PolygonZone] :
    """
    Returns warning and critical zone coordinates as Numpy.ndarrays and also as Supervision Polygon zones
    """
    try:
        y1 = int(0.80 * orig_height)
        y2 = int(0.99 * orig_height)
        y3 = int(y1 - 0.05 * orig_height)
        x1 = int(0.45 * orig_width)
        x2 = int(0.55 * orig_width) 
        x3 = int(0.7 * orig_width) 
        x4 = int(0.3 * orig_width)
        c1 = [x1, y1]
        c2 = [x2, y1]
        c3 = [x3, y2]
        c4 = [x4, y2]
        x5 = int(x1 - 0.25 * orig_width) 
        x6 = int(x2 + 0.25 * orig_width) 
        x7 = int(x3 + 0.25 * orig_width)  
        x8 = int(x4 - 0.25 * orig_width) 
        c5 = [x5, y3]
        c6 = [x6, y3]
        c7 = [x7, y2]
        c8 = [x8, y2]
        crit_zone_polygon_coords = np.array([c1, c2, c3, c4])
        warn_zone_polygon_coords = np.array([c5, c6, c7, c8])
        crit_zone = sv.PolygonZone(polygon = crit_zone_polygon_coords, frame_resolution_wh = (orig_width, orig_height))
        warn_zone = sv.PolygonZone(polygon = warn_zone_polygon_coords, frame_resolution_wh = (orig_width, orig_height))
        lg.info(f"get_wc_zone_coordinates() - Zone Coords CZ{[c1, c2, c3, c4]}, WZ{[c5, c6, c7, c8]}")
        return crit_zone_polygon_coords, warn_zone_polygon_coords, crit_zone, warn_zone
    except Exception as e:
        lg.warning(f"get_wc_zone_coordinates() - {e}")
