from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import cv2
import time
from pytesseract import image_to_string
from utils.misc_utils import parse_anchors
from utils.nms_utils import gpu_nms
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize
from model import yolov3

anchors = parse_anchors("./data/yolo_anchors.txt")
restore_path="./data/custom/yolov3.ckpt"
new_size=416

vid = cv2.VideoCapture(0)
video_frame_cnt = int(vid.get(7))
video_width = int(vid.get(3))
video_height = int(vid.get(4))
prev_frame_time = 0
new_frame_time = 0
 
with tf.compat.v1.Session() as sess:
    input_data = tf.compat.v1.placeholder(tf.float32, [1, new_size, new_size, 3], name='input_data')
    yolo_model = yolov3(1, anchors)
    times = []
    with tf.compat.v1.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, 1, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
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
            x0=int(x0)
            x1=int(x1)
            y0=int(y0)
            y1=int(y1)
            crop=img_ori[y0:y1,x0:x1]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lpText=image_to_string(crop,lang='eng',config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVYZ0123456789')
            plot_one_box(img_ori, [x0, y0, x1, y1], label=lpText + ', {:.2f}%'.format(scores_[i] * 100), color=[0,0,255])

        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms

        cv2.putText(img_ori, '{:.2f}ms - {:.1f} FPS'.format((end_time - start_time) * 1000,fps), (40, 40), 0,
                    fontScale=1, color=(255, 255, 0), thickness=2)
        cv2.imshow('image', img_ori)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()

