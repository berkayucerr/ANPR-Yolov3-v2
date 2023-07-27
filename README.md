# ANPR-Yolov3-v2
An automatic number plate recognition project coded based on the Yolo-Tensorflow Algorithm
That's running on the GPU!


---Pre-trained weights for license plate recognition---

https://drive.google.com/file/d/1JyHaj72pyC2AhIhthV2MzYtqOrY0_yyC/view?usp=sharing

1: Firstly, you have to download pre-trained weights to the main directory of project.

2: Convert pre-trained weights using convert_weights.py file and move converted weight files into data/custom directory.

3: as you working on realtime(webcam), you can use webcam.py or if you working on images, you can use image.py file.
----

Test Device M1 Macbook Pro : Avarage 17.8 fps on GPU

| Files      | Description                    |
| ------------- | ------------------------------ |
| ` image.py` | Prediction for images          |
| ` webcam.py`   | Prediction with webcam.        |

----

| Files      | FPS(Avarage)                    |
| ------------- | ------------------------------ |
| M1 MacBook Pro(GPU)  | 17.8 fps          |

![test image](https://github.com/berkayucerr/ANPR-Yolov3-v2/blob/main/Ekran%20Resmi%202022-04-18%2015.54.22.png?raw=true)
![test image](https://github.com/berkayucerr/ANPR-Yolov3-v2/blob/main/Ekran%20Resmi%202022-04-18%2015.54.41.png?raw=true)
![test image](https://github.com/berkayucerr/ANPR-Yolov3-v2/blob/main/Ekran%20Resmi%202022-04-18%2015.54.58.png?raw=true)
