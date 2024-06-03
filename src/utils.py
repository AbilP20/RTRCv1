from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import random
from typing import Tuple, List, Union
from paths import *
from logger import *

setup_logging()
lg = logging.getLogger(__name__)

class DataLoader:
    def load_models(self) -> Tuple[YOLO, YOLO, YOLO, dict]:
        """
        Return the 3 YOLO models and a modified class names dictionary of detection model
        """
        try:
            model_bbox = YOLO(MODEL_DET) # detection model
            model_mask = YOLO(MODEL_SEG) # segmentation model
            model_classify = YOLO(MODEL_CLS) # classification model
            bbox_class_names = model_bbox.names
            bbox_class_names[17] = '20'
            lg.info(f"DataLoader().load_models - models loaded")
            return model_bbox, model_mask, model_classify, bbox_class_names
        except FileNotFoundError:
            lg.warning("DataLoader().load_models - model file not found")
        except Exception as e:
            lg.warning(f"DataLoader().load_models - {e}")
        
    def load_hud_icons(self) -> Tuple[np.ndarray]:
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
            png_red_light = cv2.imread(PNG_TRAFFIC_LIGHT, cv2.IMREAD_UNCHANGED)
            png_no_entry = cv2.imread(PNG_NO_ENTRY, cv2.IMREAD_UNCHANGED)
            png_car_vicinity = cv2.imread(PNG_CAR_VICINITY, cv2.IMREAD_UNCHANGED)
            png_car_closeby = cv2.imread(PNG_CAR_CLOSEBY, cv2.IMREAD_UNCHANGED)
            png_person_vicinity = cv2.imread(PNG_PERSON_VICINITY, cv2.IMREAD_UNCHANGED)
            png_person_closeby = cv2.imread(PNG_PERSON_CLOSEBY, cv2.IMREAD_UNCHANGED)
            png_road = cv2.imread(PNG_ROAD, cv2.IMREAD_UNCHANGED)
            hud_frame = cv2.imread(HUD_BG)
            lg.info("DataLoader().load_hud_icons - hud icons loaded")
            return (png_20, png_30, png_40, png_50, png_60, png_70, png_80, png_100, png_120, 
                    png_stop, png_crosswalk, png_priority_road, png_speed_hump, png_yield, 
                    png_red_light, png_no_entry, png_car_vicinity, png_car_closeby, png_person_vicinity, 
                    png_person_closeby, png_road, hud_frame)
        except FileNotFoundError as e:
            lg.warning("DataLoader().load_hud_icons - hud image file not found")
        except Exception as e:
            lg.warning(f"DataLoader().load_hud_icons - {e}")

    def load_vid_list(self, source: Union[int, str]) -> Tuple[List, int]:
        """
        Returns a list containing the paths of test videos and the index number of the test video,
        default index = 0 i.e., the first video in data/vids will be used for testing.
        If source is a video path, then index, by default = 0
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
                    lg.info("DataLoader().load_vid_list - source video list loaded")
                return vids, vid_index
            except IndexError:
                lg.warning("DataLoader().load_vid_list - video index out of range")
            except ValueError:
                vids.append(source)
                vid_index = 0
                lg.info("DataLoader().load_vid_list - source video list loaded")
                return vids, vid_index
        except FileNotFoundError:
            lg.warning("DataLoader().load_vid_list - source video path not found")
        except Exception as e:
            lg.warning(f"DataLoader().load_vid_list - {e}")   


class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.width, self.height, self.fps, self.total_frames = sv.VideoInfo.from_video_path(video_path).__dict__.values()
        lg.info(f"VideoProcessor() - initialized test video - res{self.width,self.height}, fps({self.fps}), frames({self.total_frames})")
    
    def generate_frames(self, start_frame: int, end_frame: int):
        """
        Returns a generator of test video frames from specified start frame to the specified end_frame
        """
        try:
            frame_gen = sv.get_video_frames_generator(source_path = self.video_path, start = start_frame, end = end_frame)
            lg.info("VideoProcessor().generate_frames - test video frame generator created")
            return frame_gen
        except Exception as e:
            lg.warning(f"VideoProcessor().generate_frames - failed to create test video frame generator - {e}")

    def get_wc_zone_coordinates(self, y1:float = 0.8) -> Tuple[np.ndarray, np.ndarray,
                                               sv.detection.tools.polygon_zone.PolygonZone,
                                               sv.detection.tools.polygon_zone.PolygonZone] :
        """
        Based on the resolution of the test video, this function returns the coordinates of the warning and
        critical zone as numpy.ndarrays along with their corresponding supervision Polygon Zones
        
        y1 : float - y1 coordinate defines the height of critical zone, based on which warn-zone height will automatically be set. 

        In general cases, 0.6 <= y1 <= 0.85; range (float) = [0,1]  

        (Currently, only height has been set as a variable, as camera can be positioned at any height and probably at some angle vertically,
        which gives rise to various camera perspective, so y1 needs to be changed for every different camera,
        to get correct critical-zone and warn-zone predictions 
        """
        try:
            y1 = int(y1 * self.height)
            y2 = int(0.99 * self.height)
            y3 = int(y1 - 0.05 * self.height)
            x1 = int(0.45 * self.width)
            x2 = int(0.55 * self.width) 
            x3 = int(0.7 * self.width) 
            x4 = int(0.3 * self.width)
            c1 = [x1, y1]
            c2 = [x2, y1]
            c3 = [x3, y2]
            c4 = [x4, y2]
            x5 = int(x1 - 0.25 * self.width) 
            x6 = int(x2 + 0.25 * self.width) 
            x7 = int(x3 + 0.25 * self.width)  
            x8 = int(x4 - 0.25 * self.width) 
            c5 = [x5, y3]
            c6 = [x6, y3]
            c7 = [x7, y2]
            c8 = [x8, y2]
            crit_zone_polygon_coords = np.array([c1, c2, c3, c4])
            warn_zone_polygon_coords = np.array([c5, c6, c7, c8])
            crit_zone = sv.PolygonZone(polygon = crit_zone_polygon_coords, frame_resolution_wh = (self.width, self.height))
            warn_zone = sv.PolygonZone(polygon = warn_zone_polygon_coords, frame_resolution_wh = (self.width, self.height))
            lg.info(f"VideoProcessor().get_wc_zone_coordinates - zone coords CZ{[c1, c2, c3, c4]}, WZ{[c5, c6, c7, c8]}")
            return crit_zone_polygon_coords, warn_zone_polygon_coords, crit_zone, warn_zone
        except Exception as e:
            lg.warning(f"VideoProcessor().get_wc_zone_coordinates - {e}")

    def load_writer(self, hud_width: int, hud_height: int, fourcc) -> cv2.VideoWriter:
        """
        Returns : Inference_Video_Writer (to write moodel inferences),
                  HUD_Video_Writer (to write HUD)
        """
        try:
            vidWriter1 = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, self.fps, (self.width, self.height)) # Inference Video Writer
            vidWriter2 = cv2.VideoWriter(HUD_OUTPUT_PATH, fourcc, self.fps, (hud_width, hud_height)) # HUD Video Writer
            lg.info("VideoProcessor().load_writer - video writers loaded")
            return vidWriter1, vidWriter2
        except Exception as e:
            lg.warning(f"VideoProcessor().load_writer - failed to load video writers - {e}")

    def load_tracker(self):
        """
        Defines the trackers that tracks objects over multiple frames by assigning tracking ids to object
        """
        try:
            bbox_tracker = sv.ByteTrack(frame_rate = self.fps)
            mask_tracker = sv.ByteTrack(frame_rate = self.fps)
            lg.info("VideoProcessor().load_tracker - tracker loaded")
            return bbox_tracker, mask_tracker
        except Exception as e:
            lg.warning(f"VideoProcessor().load_tracker - failed to load trackers - {e}")
  
    def get_cur_speed(self) -> int:
        speeds = [20, 35, 40, 50, 65, 70, 80, 95, 115]
        cur_speed = random.choice(speeds)
        return cur_speed

class FrameProcessor:
    def load_mappers(self):
        """
        Returns:
            classify_det_map - maps classes of classification model to classes of detection model
            speed_limit_map - maps speed limits to the corresponding image 
        """
        classify_det_map = {0:0, 1:1, 2:17, 3:2, 4:4, 5:5, 6:6, 7:7} # cls-to-det map
        speed_limit_map = {100:PNG_100, 120:PNG_120, 20:PNG_20, 30:PNG_30, 40:PNG_40, 50:PNG_50, 60:PNG_60, 70:PNG_70, 80:PNG_80}
        lg.info("FrameProcessor().load_mappers - loaded mappers")
        return classify_det_map, speed_limit_map
    
    def load_annotators(self):
        """
        Returns different annotator objects 
        """
        ellipse_annotator = sv.EllipseAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        bbox_annotator = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.CLASS, thickness = 1)     
        color_annotator = sv.ColorAnnotator(opacity = 0.25)
        lg.info("FrameProcessor().load_mappers - loaded annotators")
        return ellipse_annotator, label_annotator, mask_annotator, bbox_annotator, color_annotator