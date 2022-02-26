from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights

num_class = 1
img_size = 416
weight_path = './data/darknet_weights/yolov3.weights'
save_path = './data/darknet_weights/yolov3.ckpt'
anchors = parse_anchors('./data/yolo_anchors.txt')

model = yolov3(num_class, anchors)
with tf.compat.v1.Session() as sess:
    inputs = tf.compat.v1.placeholder(tf.float32, [1, img_size, img_size, 3])

    with tf.compat.v1.variable_scope('yolov3'):
        feature_map = model.forward(inputs)

    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.compat.v1.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



