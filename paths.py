import os

# Current Working Directory
CUR_DIR = os.getcwd()

# HUD Icons
PNG_DIR = os.path.join( CUR_DIR,'data','images' )
PNG_20 = os.path.join( PNG_DIR, 'png_20' + '.png' )
PNG_30 = os.path.join( PNG_DIR, 'png_30' + '.png' )
PNG_40 = os.path.join( PNG_DIR, 'png_40' + '.png' )
PNG_50 = os.path.join( PNG_DIR, 'png_50' + '.png' )
PNG_60 = os.path.join( PNG_DIR, 'png_60' + '.png' )
PNG_70 = os.path.join( PNG_DIR, 'png_70' + '.png' )
PNG_80 = os.path.join( PNG_DIR, 'png_80' + '.png' )
PNG_100 = os.path.join(PNG_DIR, 'png_100' + '.png')
PNG_120 = os.path.join(PNG_DIR, 'png_120' + '.png')
PNG_STOP = os.path.join(PNG_DIR, 'png_stop' + '.png')
PNG_CROSSWALK = os.path.join(PNG_DIR, 'png_crosswalk' + '.png')
PNG_PRIORITY_ROAD = os.path.join(PNG_DIR, 'png_priority_road' + '.png')
PNG_SPEED_HUMP = os.path.join(PNG_DIR, 'png_speed_hump' + '.png')
PNG_YIELD = os.path.join(PNG_DIR, 'png_yield' + '.png')
PNG_RED_LIGHT = os.path.join(PNG_DIR, 'png_red_light1' + '.png')
PNG_NO_ENTRY  = os.path.join(PNG_DIR, 'png_no_entry' + '.png')
PNG_CAR_VICINITY = os.path.join(PNG_DIR, 'png_car_vicinity' + '.png')
PNG_CAR_CLOSEBY = os.path.join(PNG_DIR, 'png_car_closeby' + '.png')
PNG_PERSON_VICINITY = os.path.join(PNG_DIR, 'png_person_vicinity' + '.png')
PNG_PERSON_CLOSEBY  = os.path.join(PNG_DIR, 'png_person_closeby' + '.png')
PNG_ROAD = os.path.join(PNG_DIR, 'png_road' + '.png')
HUD_BG = os.path.join(PNG_DIR, 'bg' + '.png')

# Model Directory
MODEL_DIR = os.path.join(CUR_DIR,'data', 'models')

# Video Directory
VID_PATH = os.path.join(CUR_DIR,'data', 'vids')

# Output Directory
MEDIA_OUT_PATH = os.path.join(CUR_DIR, 'runs')
VIDEO_OUTPUT_PATH = os.path.join(MEDIA_OUT_PATH, 'Inference.mp4')
HUD_OUTPUT_PATH = os.path.join(MEDIA_OUT_PATH, 'HUD.mp4')

# Log File Path
LOG_FILE_PATH = os.path.join(CUR_DIR, 'logs', 'run_log.log')

# Save Predictions Path
SAVE_PRED_PATH = os.path.join(CUR_DIR, 'runs', 'preds.txt')