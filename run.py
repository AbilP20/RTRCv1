"""
Run this script in the console to see the detection, segmentation and classification capabilities of YOLO algorithm.
The classes described in readme will be detected and I have demonstrated HUD visualization using Supervision

Usage - :
    $ python run.py --src vid.mp4     # any video path
                        0,1,2,...   # put a group of videos inside data/vids and then specify the index number within the range

Usage - other args:
    $ python run.py --src 0 --device 0 --conf 0.5 --saves 1 --show 1

        --device: 0,1,2,...(cuda device id) or cpu; default = 0
        --conf: confidence parameter for the YOLO models; range=[0,1]; default = 0.5
        --saves: 1 or True will save the predictions in runs/preds.txt; default = 0
        --show: 1 or True will print the predictions on the console; default = 0                                                    
"""

# IMPORTS
import cv2
import numpy as np
import supervision as sv
import argparse
from time import time

from logger import *
from paths import *
import utils

def run(source, device, conf, saves, show):
    """
    Predicts on all frames, at the same time creates HUD for each frame and in the end, 
    saves the model inference and HUD in 2 separate .mp4 files
    """
    lg = logging.getLogger(__name__)
    lg.info("-----------------------------------------RUN START-----------------------------------------")

    # Model Load
    try:
        model_bbox, model_mask, model_classify, bbox_class_names = utils.load_models()
        print("Model loaded.")
    except Exception as e:
        print("Model failed to load. Exiting...")
        return
    
    # HUD Icon Load
    try:
        png_20, png_30, png_40, png_50, png_60, png_70, png_80, png_100, png_120,\
        png_stop, png_crosswalk, png_priority_road, png_speed_hump, png_yield,\
        png_red_light, png_no_entry, png_car_vicinity, png_car_closeby, png_person_vicinity,\
        png_person_closeby, png_road, hud_frame = utils.load_hud_icons()
        road_height, road_width, _ = png_road.shape
        hud_height, hud_width, _ = hud_frame.shape
    except Exception as e:
        print("HUD icon failed to load. Exiting...")
        return
    
    # Videos load
    try:
        vids, vid_index = utils.load_vid_list(source)
    except Exception as e:
        lg.warning("Exiting...")
        print("Source video list failed to load. Exiting...")
        return 
    
    try:
        # Bbox mapping and CurSpeed-SpeedHudIcon mapping
        classify_det_map = {0:0, 1:1, 2:17, 3:2, 4:4, 5:5, 6:6, 7:7}
        speed_limit_map = {100:PNG_100, 120:PNG_120, 20:PNG_20, 30:PNG_30, 40:PNG_40, 50:PNG_50, 60:PNG_60, 70:PNG_70, 80:PNG_80}

        # Annotators
        ellipse_annotator = sv.EllipseAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        bbox_annotator = sv.BoundingBoxAnnotator(color_lookup = sv.ColorLookup.CLASS, thickness = 1)     
        color_annotator = sv.ColorAnnotator(opacity = 0.25)

        # Video info
        try:
            vid_src = vids[vid_index]
            lg.info("Test video loaded.")
            print("Test video loaded.")
        except Exception as e:
            lg.warning("Video Index not found. Exiting...")
            print("Given video could not be loaded. Exiting...")
            return 
        
        orig_width, orig_height, orig_fps, orig_total_frames = utils.get_vid_info(vid_src)
        crit_zone_coords, warn_zone_coords, crit_zone, warn_zone = utils.get_wc_zone_coordinates(orig_width, orig_height)
        crit_zone_polygon_annotator = sv.PolygonZoneAnnotator(crit_zone,sv.Color.YELLOW,text_color=sv.Color.BLACK)
        warn_zone_polygon_annotator = sv.PolygonZoneAnnotator(warn_zone,sv.Color.GREEN,text_color=sv.Color.BLACK)

        # Speed Variables
        cur_speed_counter = 0
        cur_speed = utils.get_cur_speed()
        cur_speed_limit = 60

        # New Width and Height
        new_width = 1200
        new_height = int(new_width * (orig_height / orig_width))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # cv2 Video Writers
        inference_vid_writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, orig_fps, (orig_width, orig_height))
        hud_vid_writer = cv2.VideoWriter(HUD_OUTPUT_PATH, fourcc, orig_fps, (hud_width, hud_height))
    
        # Trackers
        bbox_tracker = sv.ByteTrack(frame_rate = orig_fps)
        mask_tracker = sv.ByteTrack(frame_rate = orig_fps)

        lg.info("Loaded mappers, annotators, test video info, zone coordinates, variables, video writers, trackers.")
    except Exception as e:
        lg.critical(f"Failed to load critical variables - {e}")
        lg.warning("Exiting...")
        print("Failed to load critical variables. Exiting...")
        return

    # MAIN
    frame_start = 0
    frame_end = orig_total_frames
    frame_count = 0

    total_inference_time = 0
    try:
        lg.info("Video inference and HUD creation started...")
        print("Inference and HUD creation started.\n")
        video_frames = sv.get_video_frames_generator(source_path = vid_src, start = frame_start, end = frame_end)
        for frame in video_frames:
            frame_count += 1
            cur_speed_counter += 1 
            if cur_speed_counter == 90:
                cur_speed = utils.get_cur_speed()
                cur_speed_counter = 0

            # DETECTION PREDS
            model_time_start = time()
            results_bbox = model_bbox(frame, conf = conf, device = device, classes = [i for i in range(17) if i not in [10,6,4]], verbose=False)[0]
            model_time_end = time()
            total_inference_time += (model_time_end - model_time_start)
            detections_bbox = sv.Detections.from_ultralytics(results_bbox)

            # CLASSIFICATION PREDS
            modified_classids = []
            for object in detections_bbox:
                if object[3] in [0, 1, 2, 4, 5, 6, 7]:
                    x_min, y_min, x_max, y_max = map(int, object[0].tolist())
                    cropped_frame = frame[y_min:y_max, x_min:x_max]
                    model_time_start = time()
                    results_classify = model_classify(cropped_frame, device = device, verbose=False)[0]
                    model_time_end = time()
                    total_inference_time += (model_time_end - model_time_start)
                    classifications = sv.Classifications.from_ultralytics(results_classify)
                    sl_list = list(classifications.confidence[0:3]) + list(classifications.confidence[4:8])
                    max_prob_index = sl_list.index(max(sl_list))
                    if max_prob_index > 2:
                        max_prob_index += 1
                    max_prob_classid = classifications.class_id[max_prob_index]
                    mapped_det_classid = classify_det_map[max_prob_classid]
                    modified_classids.append(mapped_det_classid)
                else:
                    modified_classids.append(object[3])
            detections_bbox.class_id = np.array(modified_classids)
            detections_bbox = bbox_tracker.update_with_detections(detections_bbox)
            labels_bbox = [f"#{tracker_id} {bbox_class_names[class_id]}" for tracker_id,class_id in zip(detections_bbox.tracker_id,detections_bbox.class_id)]
            bbox_anno_frame = bbox_annotator.annotate(scene = frame.copy(), detections = detections_bbox)  
            bbox_anno_frame = color_annotator.annotate(scene = bbox_anno_frame, detections = detections_bbox)    
            bbox_anno_frame = label_annotator.annotate(scene = bbox_anno_frame, detections = detections_bbox, labels = labels_bbox)   
            
            # MASK PREDS
            model_time_start = time()
            results_masks = model_mask(frame, conf = conf, device = device, verbose=False)[0]
            model_time_end = time()
            total_inference_time += (model_time_end - model_time_start)

            detections_masks = sv.Detections.from_ultralytics(results_masks)
            mask_anno_frame = mask_annotator.annotate(scene = bbox_anno_frame, detections = detections_masks)
            detections_masks = mask_tracker.update_with_detections(detections_masks)
            labels_masks = [f"#{tracker_id} person" for tracker_id in detections_masks.tracker_id]
            mask_anno_frame = ellipse_annotator.annotate(scene = mask_anno_frame, detections = detections_masks)
            mask_anno_frame = bbox_annotator.annotate(scene = mask_anno_frame, detections = detections_masks)
            mask_anno_frame = label_annotator.annotate(scene = mask_anno_frame, detections = detections_masks, labels = labels_masks)

            # ZONE COUNTS
            crit_zone_person_count_list = list(crit_zone.trigger(detections = detections_masks))
            crit_zone_person_count = crit_zone_person_count_list.count(True)
            warn_zone_person_count_list = list(warn_zone.trigger(detections = detections_masks))
            warn_zone_person_count = warn_zone_person_count_list.count(True) - crit_zone_person_count

            crit_zone_car_count_list = list(crit_zone.trigger(detections = detections_bbox[detections_bbox.class_id == 15]))
            crit_zone_car_count = crit_zone_car_count_list.count(True)
            warn_zone_car_count_list = list(warn_zone.trigger(detections = detections_bbox[detections_bbox.class_id == 15]))
            warn_zone_car_count = warn_zone_car_count_list.count(True) - crit_zone_car_count

            count_vic_cb_frame = sv.draw_filled_rectangle(scene = mask_anno_frame, rect = sv.Rect(10,10,100,100), color = sv.Color.WHITE)    
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = 'In-Vicinity',text_anchor = sv.Point(60,30),text_color=sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Person : {warn_zone_person_count}", text_anchor = sv.Point(60,55), text_color = sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Vehicles : {warn_zone_car_count}", text_anchor = sv.Point(60,80), text_color = sv.Color.BLACK)

            count_vic_cb_frame = sv.draw_filled_rectangle(scene = count_vic_cb_frame,rect = sv.Rect(150,10,100,100),color=sv.Color.WHITE)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = 'Close-by',text_anchor=sv.Point(200,30),text_color=sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Person : {crit_zone_person_count}", text_anchor=sv.Point(200,55), text_color=sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Vehicles : {crit_zone_car_count}", text_anchor=sv.Point(200,80), text_color=sv.Color.BLACK)
            
            # ZONE DRAWS
            zone_draw_frame = sv.draw_line(scene=count_vic_cb_frame, start = sv.Point(crit_zone_coords[0][0], crit_zone_coords[0][1]), end = sv.Point(crit_zone_coords[1][0], crit_zone_coords[1][1]), color = sv.Color(255,126,87))
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(crit_zone_coords[1][0], crit_zone_coords[1][1]), end = sv.Point(crit_zone_coords[2][0], crit_zone_coords[2][1]), color = sv.Color(255,126,87))
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(crit_zone_coords[0][0], crit_zone_coords[0][1]), end = sv.Point(crit_zone_coords[3][0], crit_zone_coords[3][1]), color = sv.Color(255,126,87))

            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(warn_zone_coords[0][0], warn_zone_coords[0][1]), end = sv.Point(warn_zone_coords[3][0], warn_zone_coords[3][1]), color = sv.Color(255,239,97))
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(warn_zone_coords[1][0], warn_zone_coords[1][1]), end = sv.Point(warn_zone_coords[2][0], warn_zone_coords[2][1]), color = sv.Color(255,239,97))

            # HUD IMPLEMENTATION 
            sign_height, sign_width = int(0.075*hud_height), int(0.075*hud_height)
            road_img_height = int(0.2*hud_height)
            road_img_width = int(road_width/road_height*road_img_height) 

            hud_frame1 = sv.draw_image(scene = hud_frame.copy(), image = png_road, opacity=0.9, rect=sv.Rect(854, 700, road_img_width, road_img_height))
            hud_frame1 = sv.draw_image(scene = hud_frame1, image = speed_limit_map[cur_speed_limit], opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
            
            # Car Sign
            if crit_zone_car_count > 0:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_car_closeby, opacity=0.9, rect=sv.Rect(1610, 750, sign_width, sign_height))
            elif warn_zone_car_count>0:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_car_vicinity, opacity=0.9, rect=sv.Rect(1610, 750, sign_width, sign_height))
            # Person Sign
            if crit_zone_person_count > 0:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_person_closeby, opacity=0.9, rect=sv.Rect(1470, 755, 65, sign_height-10))
            elif warn_zone_person_count>0:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_person_vicinity, opacity=0.9, rect=sv.Rect(1470, 755, 65, sign_height-10))
            # Speed Limit Signs
            if 0 in detections_bbox.class_id: # SL-100
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_100, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 100
            elif 1 in detections_bbox.class_id: # SL-120
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_120, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 120
            elif 2 in detections_bbox.class_id: # SL-30
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_30, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 30
            elif 3 in detections_bbox.class_id: # SL-40
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_40, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 40
            elif 4 in detections_bbox.class_id: # SL-50
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_50, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 50
            elif 5 in detections_bbox.class_id: # SL-60
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_60, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 60
            elif 6 in detections_bbox.class_id: # SL-70
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_70, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 70
            elif 7 in detections_bbox.class_id: # SL-80
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_80, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 80
            elif 17 in detections_bbox.class_id: # SL-20
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_20, opacity=0.9, rect=sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 20
            # Crosswalk
            if 8 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_crosswalk, opacity=0.9, rect=sv.Rect(1400, 550, sign_width, sign_height))
            # No-Entry
            if 9 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_no_entry, opacity=1, rect=sv.Rect(700, 550, sign_width, sign_height))
            # Priority Road
            if 11 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_priority_road, opacity=0.9, rect=sv.Rect(1610, 550, sign_width, sign_height))
            # Speed Hump
            if 12 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_speed_hump, opacity=1, rect=sv.Rect(685, 930, sign_width+25, sign_height+25))
            # Stop
            if 13 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_stop, opacity=0.9, rect=sv.Rect(920, 550, sign_width, sign_height))
            # Red-Light
            if 14 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_red_light, opacity=0.9, rect=sv.Rect(1155, 535, sign_width+10, sign_height+10))
            # Yield    
            if 16 in detections_bbox.class_id:
                hud_frame1 = sv.draw_image(scene = hud_frame1, image = png_yield, opacity=0.9, rect=sv.Rect(700, 750, sign_width, sign_height))

            hud_frame1 = sv.draw_text(hud_frame1, f"{cur_speed} Kmph", sv.Point(1230,1120), sv.Color.GREEN,2,5)
            if cur_speed>cur_speed_limit:
                hud_frame1 = sv.draw_text(hud_frame1, f"OVERSPEED!", sv.Point(1230,1220), sv.Color.RED,2,5)
            progress = frame_count/(frame_end-frame_start)*100
            if (show in ['1','True']) or (saves in ['1','True']):
                class_ids = detections_bbox.tracker_id.tolist()
                class_names = [bbox_class_names[class_id] for class_id in detections_bbox.class_id]
                class_confidence = [round(conf,4) for conf in detections_bbox.confidence]
                class_xyxy = [list(i) for i in detections_bbox.xyxy.round(2)]
                class_ids.extend(detections_masks.tracker_id.tolist())
                class_names.extend(['pedestrian' for i in range(len(detections_masks.class_id))])
                class_confidence.extend([round(conf,4) for conf in detections_masks.confidence])
                class_xyxy.extend([list(i) for i in detections_masks.xyxy.round(2)])
                d1 = {'id': class_ids, 'class':class_names, 'conf': class_confidence, 'xyxy': class_xyxy}
            if (saves in ['1','True']):
                if frame_count==1:
                    with open(SAVE_PRED_PATH,'w') as f:                
                        f.write(f"Video Source: {vid_src}\nResolution: {orig_width,orig_height}\nFrames: {orig_total_frames}\nFPS: {orig_fps}\n\n")
                        f.write(f'Frame {frame_count}/{(frame_end-frame_start)}:\n{d1}')
                        f.write("\n")
                else:
                    with open(SAVE_PRED_PATH,'a') as f:
                        f.write(f'Frame {frame_count}/{(frame_end-frame_start)}:\n{d1}')
                        f.write("\n")
            else:
                with open(SAVE_PRED_PATH,'w') as f:
                    pass
            if (show in ['1','True']):
                print(f'Frame {frame_count}/{(frame_end-frame_start)} - {d1}')
            else:
                print(f'Processed: {frame_count}/{(frame_end-frame_start)} - {progress:.2f}%', end='\r')
            hud_vid_writer.write(hud_frame1)
            inference_vid_writer.write(zone_draw_frame)
        lg.info("Detection and HUD creation successful.")
        inference_vid_writer.release()
        hud_vid_writer.release()
        vid_inference_time = round(total_inference_time,2)
        model_speed = round(frame_count/total_inference_time,2)
        lg.info(f"{frame_count}/{frame_end-frame_start} frames processed.")
        lg.info(f"Inference Time - {vid_inference_time}s, Speed - {model_speed}fps")
        lg.info("------------------------------------------RUN END------------------------------------------")
        print()
        print(f"Inference Time: {vid_inference_time}s, Speed: {model_speed}fps")
    except KeyboardInterrupt:
        lg.info("Keyboard Interrupt.")
        print()
        print("Process interrupted. Exiting...")
        print("")
    except Exception as e:
        lg.info(f"{frame_count}/{frame_end - frame_start} frames processed.")
        lg.warning(f"Detection Failed - {e}")
        print(f"Detection Failed after processing {frame_count}/{frame_end-frame_start} frames. Exiting...")
        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default = 0, help = "video source path(str) outside the dir or video index number within the data/vids dir")
    parser.add_argument('--device', default = 0, help = "device id for inference (cuda device = 0, 1, 2,.. or cpu)")
    parser.add_argument('--conf', type = float, default = 0.5, help = "confidence value for all predictions (range = [0,1])")
    parser.add_argument('--saves', default = 0, help = "saves predictions to runs/preds.txt")
    parser.add_argument('--show', default = 0, help = "shows predictions on the console")
    args = parser.parse_args()

    run(args.src, args.device, args.conf, args.saves, args.show)