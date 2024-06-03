"""
Usage - :
    $ python src/run.py --source path_to_vid.mp4     # any video path
                          0,1,2,...   # put a group of videos inside data/vids and then specify the index number within the range
Usage - other args:
    $ python src/run.py --source 0 --device 0 --conf 0.5 --start 200 --end -1 --save 1 --show 1

        --device: 0,1,2,..(cuda device id) or cpu; default = 0
        --conf: confidence parameter for the YOLO models; range (float) = [0,1]; default = 0.5
        --start: start frame no. to start the predictions from; default = 0, i.e start from the beginning 
        --end: end frame no. to stop the predictions at; default = -1, i.e till the last available frame
        --save: 1 or True will save the predictions in runs/preds.txt; default = 0
        --show: 1 or True will print the predictions on the console; default = 0                                                    
"""

# IMPORTS
import cv2
import numpy as np
import supervision as sv
import argparse
from time import time
from logger import *
from paths import SAVE_PRED_PATH
from utils import DataLoader, VideoProcessor, FrameProcessor

def run(source, device, conf, start_f, end_f, save, show):

    setup_logging()
    lg = logging.getLogger(__name__)
    lg.info("-----------------------------------------RUN START-----------------------------------------")

    # DATA LOAD
    print("Loading data: models, hud icons and video list")
    try:
        model_bbox, model_mask, model_classify, bbox_class_names = DataLoader().load_models() # models
        png_20, png_30, png_40, png_50, png_60, png_70, png_80, png_100, png_120,\
        png_stop, png_crosswalk, png_priority_road, png_speed_hump, png_yield,\
        png_red_light, png_no_entry, png_car_vicinity, png_car_closeby, png_person_vicinity,\
        png_person_closeby, png_road, hud_frame = DataLoader().load_hud_icons() # hud icons
        road_height, road_width, _ = png_road.shape
        hud_height, hud_width, _ = hud_frame.shape
        vids, vid_index = DataLoader().load_vid_list(source) # video list
    except Exception as e:
        print("Failed to load data. Exiting...")
        return
    
    # DEFINING TEST VIDEO VARIABLES 
    try:
        vid_src = vids[vid_index]
        testVideo = VideoProcessor(vid_src)
        orig_width, orig_height, orig_fps, orig_total_frames = testVideo.width, testVideo.height, testVideo.fps, testVideo.total_frames
        crit_zone_coords, warn_zone_coords, crit_zone, warn_zone = testVideo.get_wc_zone_coordinates(y1 = 0.8) # critical-warn zones
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        inference_vid_writer, hud_vid_writer = testVideo.load_writer(hud_width, hud_height, fourcc) # video writers
        bbox_tracker, mask_tracker = testVideo.load_tracker() # trackers   
        lg.info("Test video and variables loaded")
        print("Test video variables loaded")
    except Exception as e:
        lg.warning("Failed to load test video variables")
        print("Failed to load test video variables. Exiting...")
        return  

    # DEFINING MORE FRAME VARIABLES
    try:
        ## Frame Range and Count 
        start_frame = int(start_f)
        if int(end_f) == -1 or int(end_f) > orig_total_frames: end_frame = orig_total_frames
        else: end_frame = int(end_f)
        n_frames = end_frame - start_frame
        frame_count = 0
        lg.info(f"start_frame: {start_frame}, end_frame: {end_frame}, frames: {n_frames}")
        classify_det_map, speed_limit_map = FrameProcessor().load_mappers()  # mappers
        ellipse_annotator, label_annotator, mask_annotator, bbox_annotator, color_annotator = FrameProcessor().load_annotators() # annotators
        ## Speed Variables
        cur_speed_counter = 0
        cur_speed = testVideo.get_cur_speed()
        cur_speed_limit = 60
    except Exception as e:
        lg.warning(f"Failed to load frame variables - {e}")
        print("Failed to load frame variables. Exiting...")
        return

    # MAIN
    total_inference_time = 0 # model inference time
    try:
        lg.info("Video inference and HUD creation started...")
        print("Inference and HUD creation started...")
        video_frames = testVideo.generate_frames(start_frame, end_frame) # frame generator
        
        for frame in video_frames:
            frame_count += 1
            cur_speed_counter += 1 
            if cur_speed_counter == 90:
                cur_speed = testVideo.get_cur_speed()
                cur_speed_counter = 0

            # PREDICTIONS
            ## Detection Pred
            model_time_start = time()
            results_bbox = model_bbox(frame, conf = conf, device = device, classes = [i for i in range(17) if i not in [10,6,4]], verbose=False)[0]
            model_time_end = time()
            total_inference_time += (model_time_end - model_time_start)
            detections_bbox = sv.Detections.from_ultralytics(results_bbox)
            ## Classification Preds
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
            ## Update Detections
            detections_bbox.class_id = np.array(modified_classids)
            detections_bbox = bbox_tracker.update_with_detections(detections_bbox)
            labels_bbox = [f"#{tracker_id} {bbox_class_names[class_id]}" for tracker_id,class_id in zip(detections_bbox.tracker_id,detections_bbox.class_id)]
            bbox_anno_frame = bbox_annotator.annotate(scene = frame.copy(), detections = detections_bbox)  
            bbox_anno_frame = color_annotator.annotate(scene = bbox_anno_frame, detections = detections_bbox)    
            bbox_anno_frame = label_annotator.annotate(scene = bbox_anno_frame, detections = detections_bbox, labels = labels_bbox)   
            # Segmentation Preds
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
            #--------------------------------------------------------------------------------------------------------------------
            # ZONE COUNTS
            crit_zone_person_count = list(crit_zone.trigger(detections = detections_masks)).count(True)
            warn_zone_person_count = list(warn_zone.trigger(detections = detections_masks)).count(True) - crit_zone_person_count
            crit_zone_car_count = list(crit_zone.trigger(detections = detections_bbox[detections_bbox.class_id == 15])).count(True)
            warn_zone_car_count = list(warn_zone.trigger(detections = detections_bbox[detections_bbox.class_id == 15])).count(True) - crit_zone_car_count
            ## In-Vicicty
            count_vic_cb_frame = sv.draw_filled_rectangle(scene = mask_anno_frame, rect = sv.Rect(10,10,100,100), color = sv.Color.WHITE)    
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = 'In-Vicinity',text_anchor = sv.Point(60,30),text_color=sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Person : {warn_zone_person_count}", text_anchor = sv.Point(60,55), text_color = sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Vehicles : {warn_zone_car_count}", text_anchor = sv.Point(60,80), text_color = sv.Color.BLACK)
            ## Closeby
            count_vic_cb_frame = sv.draw_filled_rectangle(scene = count_vic_cb_frame,rect = sv.Rect(150,10,100,100),color=sv.Color.WHITE)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = 'Close-by',text_anchor=sv.Point(200,30),text_color=sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Person : {crit_zone_person_count}", text_anchor=sv.Point(200,55), text_color=sv.Color.BLACK)
            count_vic_cb_frame = sv.draw_text(scene = count_vic_cb_frame, text = f"Vehicles : {crit_zone_car_count}", text_anchor=sv.Point(200,80), text_color=sv.Color.BLACK)
            #--------------------------------------------------------------------------------------------------------------------
            # ZONE DRAWS
            ## Critical Zone
            zone_draw_frame = sv.draw_line(scene=count_vic_cb_frame, start = sv.Point(crit_zone_coords[0][0], crit_zone_coords[0][1]), end = sv.Point(crit_zone_coords[1][0], crit_zone_coords[1][1]), color = sv.Color(255,126,87))
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(crit_zone_coords[1][0], crit_zone_coords[1][1]), end = sv.Point(crit_zone_coords[2][0], crit_zone_coords[2][1]), color = sv.Color(255,126,87))
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(crit_zone_coords[0][0], crit_zone_coords[0][1]), end = sv.Point(crit_zone_coords[3][0], crit_zone_coords[3][1]), color = sv.Color(255,126,87))
            ## Warn Zone
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(warn_zone_coords[0][0], warn_zone_coords[0][1]), end = sv.Point(warn_zone_coords[3][0], warn_zone_coords[3][1]), color = sv.Color(255,239,97))
            zone_draw_frame = sv.draw_line(scene=zone_draw_frame, start = sv.Point(warn_zone_coords[1][0], warn_zone_coords[1][1]), end = sv.Point(warn_zone_coords[2][0], warn_zone_coords[2][1]), color = sv.Color(255,239,97))
            
            inference_vid_writer.write(zone_draw_frame)
            #--------------------------------------------------------------------------------------------------------------------
            # HUD IMPLEMENTATION 
            sign_height, sign_width = int(0.075 * hud_height), int(0.075 * hud_height)
            road_img_height = int(0.2 * hud_height)
            road_img_width = int(road_width / road_height * road_img_height) 
            hud_anno_frame = sv.draw_image(scene = hud_frame.copy(), image = png_road, opacity = 0.9, rect = sv.Rect(854, 700, road_img_width, road_img_height))
            hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = speed_limit_map[cur_speed_limit], opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
            ## Car Sign
            if crit_zone_car_count > 0:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_car_closeby, opacity = 0.9, rect = sv.Rect(1610, 750, sign_width, sign_height))
            elif warn_zone_car_count > 0:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_car_vicinity, opacity = 0.9, rect = sv.Rect(1610, 750, sign_width, sign_height))
            ## Person Sign
            if crit_zone_person_count > 0:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_person_closeby, opacity = 0.9, rect = sv.Rect(1470, 755, 65, sign_height-10))
            elif warn_zone_person_count > 0:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_person_vicinity, opacity = 0.9, rect = sv.Rect(1470, 755, 65, sign_height-10))
            ## Speed Limit Signs
            if 0 in detections_bbox.class_id: # SL-100
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_100, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 100
            elif 1 in detections_bbox.class_id: # SL-120
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_120, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 120
            elif 2 in detections_bbox.class_id: # SL-30
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_30, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 30
            elif 3 in detections_bbox.class_id: # SL-40
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_40, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 40
            elif 4 in detections_bbox.class_id: # SL-50
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_50, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 50
            elif 5 in detections_bbox.class_id: # SL-60
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_60, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 60
            elif 6 in detections_bbox.class_id: # SL-70
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_70, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 70
            elif 7 in detections_bbox.class_id: # SL-80
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_80, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 80
            elif 17 in detections_bbox.class_id: # SL-20
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_20, opacity = 0.9, rect = sv.Rect(1020, 920, sign_width, sign_height))
                cur_speed_limit = 20
            ## Crosswalk
            if 8 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_crosswalk, opacity = 0.9, rect = sv.Rect(1400, 550, sign_width, sign_height))
            ## No-Entry
            if 9 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_no_entry, opacity = 1, rect = sv.Rect(700, 550, sign_width, sign_height))
            ## Priority Road
            if 11 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_priority_road, opacity = 0.9, rect = sv.Rect(1610, 550, sign_width, sign_height))
            ## Speed Hump
            if 12 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_speed_hump, opacity = 1, rect = sv.Rect(685, 930, sign_width+25, sign_height+25))
            # Stop
            if 13 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_stop, opacity = 0.9, rect = sv.Rect(920, 550, sign_width, sign_height))
            ## Traffic-Light
            if 14 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_red_light, opacity = 0.9, rect = sv.Rect(1155, 535, sign_width+10, sign_height+10))
            ## Yield    
            if 16 in detections_bbox.class_id:
                hud_anno_frame = sv.draw_image(scene = hud_anno_frame, image = png_yield, opacity = 0.9, rect = sv.Rect(700, 750, sign_width, sign_height))
            ## Current Speed and Overspeed
            hud_anno_frame = sv.draw_text(hud_anno_frame, f"{cur_speed} Kmph", sv.Point(1230,1120), sv.Color.GREEN, 2 ,5)
            if cur_speed > cur_speed_limit:
                hud_anno_frame = sv.draw_text(hud_anno_frame, f"OVERSPEED!", sv.Point(1230,1220), sv.Color.RED, 2 ,5)
            
            hud_vid_writer.write(hud_anno_frame)
            #--------------------------------------------------------------------------------------------------------------------
            # SAVE and SHOW  
            progress = frame_count / n_frames * 100
            if (show in ['1','True']) or (save in ['1','True']): 
                class_ids = detections_bbox.tracker_id.tolist() + detections_masks.tracker_id.tolist()
                class_names = [bbox_class_names[class_id] for class_id in detections_bbox.class_id] + ['pedestrian' for i in range(len(detections_masks.class_id))]
                class_confidence = [round(conf,4) for conf in detections_bbox.confidence] + [round(conf,4) for conf in detections_masks.confidence]
                class_xyxy = [list(i) for i in detections_bbox.xyxy.round(2)] + [list(i) for i in detections_masks.xyxy.round(2)]
                d1 = {'id': class_ids, 'class':class_names, 'conf': class_confidence, 'xyxy': class_xyxy}
            if (save in ['1','True']): 
                if frame_count == 1:
                    with open(SAVE_PRED_PATH,'w') as f:                
                        f.write(f"Video Source: {vid_src}\nResolution: {orig_width,orig_height}\nTotal Frames: {orig_total_frames}\nTest Video Frames: {n_frames}\nFPS: {orig_fps}\n\n")
                        f.write(f'Frame {frame_count}/{(n_frames)}:\n{d1}')
                        f.write("\n")
                else:
                    with open(SAVE_PRED_PATH,'a') as f:
                        f.write(f'Frame {frame_count}/{(n_frames)}:\n{d1}')
                        f.write("\n")
            else:
                with open(SAVE_PRED_PATH,'w') as f:
                    pass
            if (show in ['1','True']):
                print(f'Frame {frame_count}/{n_frames}:\n{d1}')
            else:
                print(f'Processed: {frame_count}/{n_frames} - {progress:.2f}%', end='\r')
            #--------------------------------------------------------------------------------------------------------------------
        lg.info("Detection and HUD creation successful.")
        inference_vid_writer.release()
        hud_vid_writer.release()
        vid_inference_time = round(total_inference_time, 2)
        model_speed = round(frame_count / total_inference_time, 2)
        lg.info(f"{frame_count}/{n_frames} frames processed.")
        lg.info(f"Inference Time - {vid_inference_time}s, Speed - {model_speed}fps")
        lg.info("------------------------------------------RUN END------------------------------------------")
        print()
        print(f"Inference Time: {vid_inference_time}s, Speed: {model_speed}fps")
    except KeyboardInterrupt:
        lg.info("Keyboard Interrupt")
        print()
        print("Process interrupted. Exiting...")
    except Exception as e:
        lg.warning(f"detection Failed after processing {frame_count} frame(s) - {e}")
        print(f"Detection Failed after processing {frame_count}/{end_frame-start_frame} frames. Exiting...")
        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default = 0, help = "video source path(str) outside the dir or video index number within the data/vids dir")
    parser.add_argument('--device', default = 0, help = "device id for inference (cuda device = 0, 1, 2,.. or cpu)")
    parser.add_argument('--conf', type = float, default = 0.5, help = "confidence value for all predictions (range = [0,1])")
    parser.add_argument('--start', default = 0, help = "specify the frame number to start predictions from")
    parser.add_argument('--end', default = -1, help = "specify the frame number to stop predictions at")
    parser.add_argument('--save', default = 0, help = "save predictions to runs/preds.txt")
    parser.add_argument('--show', default = 0, help = "shows predictions on the console")
    
    args = parser.parse_args()

    run(args.source, args.device, args.conf, args.start, args.end, args.save, args.show)