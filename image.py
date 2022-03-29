from ctypes.wintypes import LPBYTE
from inspect import CORO_SUSPENDED
import tensorflow as tf
import cv2
import numpy as np
from utils.misc_utils import parse_anchors
from utils.nms_utils import gpu_nms
from blob_detection import blackW
from utils.plot_utils import plot_one_box
from utils.data_aug import letterbox_resize
from utils.char_rec import char_rec
from model import yolov3
import os

anchors = parse_anchors("./data/yolo_anchors.txt")
restore_path="./data/custom/yolov3.ckpt"
new_size=416
test_files_path='test_data/'
with tf.compat.v1.Session() as sess:
    input_data = tf.compat.v1.placeholder(tf.float32, [1, new_size, new_size, 3], name='input_data')
    yolo_model = yolov3(1, anchors)
    with tf.compat.v1.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    pred_scores = pred_confs * pred_probs
    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, 1, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, restore_path)
    tarama=os.scandir(test_files_path)
    for belge in tarama:
        img_ori =cv2.imread(test_files_path+belge.name)
        print('image = '+belge.name)
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size, new_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        print(boxes_)
        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            boxtemp=boxes_[0]
            x=int(boxtemp[0])
            y=int(boxtemp[1])
            w=int(boxtemp[2])
            h=int(boxtemp[3])
            crop=img_ori[y:h,x:w]
            # crop=blackW(crop)
            lpText,char_boxes=char_rec(crop)
            hImg, wImg ,_= crop.shape
            for b in char_boxes.splitlines():
                b = b.split(' ')
                print(b)
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                cv2.rectangle(crop, (x, hImg - y), (w, hImg - h), (50, 50, 255), 1)
                cv2.putText(crop, b[0], (x, hImg - y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 205, 50), 1)
                cv2.imshow('temp',crop)
                cv2.waitKey(0)
                # plot_one_box(img_ori, [a1, a2, a3, a4], label=title , color=[0,255,0])
            plot_one_box(img_ori, [x0, y0, x1, y1], label=lpText + ', {:.2f}%'.format(scores_[i] * 100), color=[0,0,255])
            print(lpText)
        cv2.imshow('image', img_ori)
        cv2.waitKey(0)