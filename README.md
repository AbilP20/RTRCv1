
<h1>Real Time Roadway Companion v1 (RTRCv1)
<img src="https://github.com/AbilP20/RTRCv1/assets/114949809/ed87fa22-4887-4a04-9a47-3b14c4b47bf3" alt="GPLv3 License" width="75"/>
</h1>
<br>

Real Time Roadway Companion (RTRC) is a real-time computer vision application in road travel assistance domain. 
It has been implemented using the widely known single-stage detection algorithm - ***YOLO*** , known for it's speed and accuracy. In this project, I have used the YOLOv8 version to train 3 models -
- **Detection model** : aims to detect vehicles and traffic signs (categories of class is displayed in detail in another topic)
- **Segmentation model** : aims to detect and segment pedestrians 
- **Classification model** : aims to classify the speed limits to refine the speed limit detection

I have also demonstrated the integration of deep learning model with Heads-Up-Display by using ***Supervision*** python library.
For training, combination of data from multiple datasets has been used based on the classes being detected. Due to training limitations, model has been trained on fewer images:
- around 2200 for detection model
- around 2500 for segmentation model
- around 12000 for classification model
<br><br>

## Dependancies

<strong> Ultralytics</strong>
<br>
<img src="https://github.com/AbilP20/RTRCv1/assets/114949809/ead2f3e2-cbfb-4620-9cd1-e4a0445f970e" alt="Supervision Banner" width="250"/>
<br>
<a href="https://github.com/ultralytics/ultralytics">Ultralytics Github</a> | <a href="https://docs.ultralytics.com/">Ultralytics Website</a>
<br><br><br>
<strong>Supervision</strong>
<br>
<img src="https://github.com/AbilP20/RTRCv1/assets/114949809/2791a89c-ab4d-4aef-9842-ed4de7c820d6" alt="Supervision Banner" width="250"/>
<br>
<a href="https://github.com/roboflow/supervision">Supervision Github</a> | <a href="https://supervision.roboflow.com/latest/">Supervision Website</a>
<br><br><br>
<strong>OpenCV-Python</strong>
<br>
<img src="https://github.com/AbilP20/RTRCv1/assets/114949809/541ae3d3-6832-4edd-a48a-e207b7db7d5f" alt="Opencv-Python Banner" width="250"/>
<br><br>

## Built With

- [Python](https://www.python.org/)
- [Roboflow](https://roboflow.com/)
- [Supervision](https://supervision.roboflow.com/latest/)
- [Ultralytics](https://docs.ultralytics.com/)
- [Numpy](https://numpy.org/)
- [Opencv-Python](https://docs.opencv.org/4.x/)
<br><br>


## Class Categories

The image below shows the classes that will be detected. These specific classes were chosen majorily based on the availabilty of train data and some based on project requirements.<br>
The top 2 rows of classes in the below image are actual traffic signs that we encounter often. The bottom row classes are custom signs that I have used for the purpose of displaying in the HUD whether the person or vehicle is nearby or in-vicinity. A simple approach approach of relative distance has been utilized to check the nearby or in-vicinity class.
<br><br>

<img src="https://github.com/AbilP20/RTRCv1/assets/114949809/f9002607-407a-4be4-b7e8-04e25ef42294" align="center" alt="Classes Image" width="300"/>
<br><br>

## Environment Setup

1. Create a directory<br>
2. Setup a virtual environment with Python v3.8
``` python
conda create -p venv python=3.8 -y
```

3. Activate venv
```python
conda activate path_to_directory/venv
```

4. It is recommended to use a GPU for better run experience. Install the appropriate pytorch version.<br> 
   Refer [Pytorch Local Install](https://pytorch.org/get-started/locally/)

5. Install the requirements.
```
pip install -r requirements.txt
```
<br>

## Usage

```python
python run.py --src 0 --device 0 --conf 0.5 --saves 0 --show 0
```
`--src` : source video path - either pass the video path or put a group of videos inside data/vids and then pass video index number; default = 0<br>
`--device` : device id - cuda device i.e. 0,1,2,.. or 'cpu'; default = 0<br>
`--conf` : confidence parameter of YOLO models - default = 0.5; range = [0, 1]<br>
`--saves` : saves the predictions inside runs/preds.txt if '1' or 'True' is passed; default = 0<br>
`--show` : displays the predictions on console if '1' or 'True' is passed; default = 0<br><br>
_NOTE: The coordinates of the 2 zones (yellow and red zone - refer the demo) needs to be changed for different camera perspectives, i.e if the same coordinates are used for every different camera perspective, then the zones may not be properly aligned with the roads and then the detections for nearby and in-vicinty vehicles and pedestrians will come out wrong._ 
<br><br>

## Results

Measuring the inference speed on RTX 3050, all 3 models combined give an average speed ranging between 20-23 fps. On Tesla T4, the average was between 28-32 fps and on Tesla P100, the average speed ranged between 37-42 fps. For testing, I tested on 480p and 720p clips from the net. This traffic signs used in this project are used in European countries, so best results will be returned if tested on European videos. Due to copyright issues on good European videos, I have not put those clips, but I have tested personally on those videos which showed good outcomes. I have shared a small demo of both, model infernce and Heads-Up-Display creation using a free-to-use video clip.
<br><br>

## Demo

- The polygon enclosing the Red lines is Critical Zone - any vehicle or pedestrian detected in this zone will be displayed in red in HUD
- The polygon enclosing the Yellow lines is Warning Zone - any vehicle or pedestrian detected in this zone will be displayed in yelloe in HUD
- All other traffic signs will be detected and showed in the HUD as and when they will be detected
- If current speed is greater than the max speed limit, then 'OVERSPEED' message also is displayed in the HUD
- The green color speed in HUD indicates current speed and is randomly generated every 3 seconds. It is just for the purpose of showing 'OVERSPEED' message

https://github.com/AbilP20/RTRCv1/assets/114949809/096ba211-deaa-47e7-b926-e7d37583332b

<br><br>

## Author

- Abil Pariyath - abilpt@gmail.com | [Github Profile](https://github.com/AbilP20)
- Project - Real Time Roadway Companion v1 | [Github Repo](https://github.com/AbilP20/RTRCv1)

All the work presented in this repo is written and executed by me, starting from data collection from various sources, separating the required data, combining them in similar classes, annotating them, training them, and finally HUD creation. No code has been copied. Due mentions have been given under Acknowledgements, regarding the official docs I referred to.
<br><br>

## License

This software is licensed under the GNU General Public License v3.0 (GPLv3). See the `LICENSE` file for more details.
<br><br>

## Contributions

All contributions are welcome. This project is a basic usage of computer vision to solve a real-world problem. There is a lot of scope, specially in the model training part as a wide number of scenarios need to be trained; classes with low number of data has to be treated properly and some classes need to be trained with negatives (as far as I could think of) so that accuracy could increase, and much more. 
<br>
This is my first complex computer vision task that I undertook. Do give a ⭐️ if you like my work!
<br><br>

## Acknowledgements

The below docs really helped me a lot. Whatever you think of performing, they will have a solution. Do check them out for any computer vision project. 
- [Ultralytics](https://docs.ultralytics.com/)
- [Roboflow Image-Preprocessing](https://docs.roboflow.com/datasets/image-preprocessing)
- [Supervision](https://supervision.roboflow.com/latest/detection/core/)
<br><br>
