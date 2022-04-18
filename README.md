# ANPR-Yolov3-v2
An anpr project coded based on the Yolo-Tensorflow Algorithm(GPU)


---Pre-trained weights for license plate recognition---

https://drive.google.com/file/d/1JyHaj72pyC2AhIhthV2MzYtqOrY0_yyC/view?usp=sharing

1: Download pre-trained weights main directory.

2: Convert pre-trained weights with convert_weights.py file and move converted weight files into data/custom directory.

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
