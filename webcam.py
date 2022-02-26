from __future__ import division, print_function
from matplotlib.pyplot import box

import tensorflow as tf
import numpy as np
import cv2
import time
from pytesseract import Output
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
import pytesseract
from skimage.segmentation import clear_border

from model import yolov3

def build_tesseract_options(psm=7):
		# tell Tesseract to only OCR alphanumeric characters
		alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		options = "-c tessedit_char_whitelist={}".format(alphanumeric)

		# set the PSM mode
		options += " --psm {}".format(psm)

		# return the built options string
		return options


anchors = parse_anchors("./data/yolo_anchors.txt")
class_names = read_class_names("./data/coco.names")
restore_path="./data/darknet_weights/yolov3.ckpt"
class_nums = 1
new_size=416
color_table =[0, 150, 0]

vid = cv2.VideoCapture(0)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
prev_frame_time = 0
new_frame_time = 0
 
with tf.compat.v1.Session() as sess:
    input_data = tf.compat.v1.placeholder(tf.float32, [1, new_size, new_size, 3], name='input_data')
    yolo_model = yolov3(class_nums, anchors)
    times = []
    with tf.compat.v1.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, class_nums, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, restore_path)

    while (True):
        ret, img_ori = vid.read()
        t1 = time.time()

        img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size, new_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        end_time = time.time()
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=class_names[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        t2 = time.time()
        if(len(boxes_)>0):
            boxtemp=boxes_[0]
            x=int(boxtemp[0])
            y=int(boxtemp[1])
            w=int(boxtemp[2])
            h=int(boxtemp[3])
            crop=img_ori[y:h,x:w]
            cv2.imshow('a',crop)
            options = build_tesseract_options(psm=7)
            lpText = pytesseract.image_to_string(crop, config=options)
            print(lpText)
        times.append(t2-t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        
        #print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))
        cv2.putText(img_ori, '{:.2f}ms - {:.1f} FPS'.format((end_time - start_time) * 1000,fps), (40, 40), 0,
                    fontScale=1, color=(255, 255, 0), thickness=2)
        cv2.imshow('image', img_ori)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()

