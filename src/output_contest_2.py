"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import cv2


def show_prediction_labels_on_image(img_path, predictions, save_path):
    """
    Shows the face recognition results visually.

    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    image = cv2.imread("img_path")

    for top, right, bottom, left in predictions:
        cv2.rectangle(image, (top, left), (bottom, right), (0, 255, 0), 3)

    cv2.save(save_path, image)


def main():
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=666)

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model("/home/thanhnn/facenet/checkpoints/20180402-114759")

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            # TODO
            # Embed test image
            print("Emb test image")
            test_path = ["/home/thanhnn/dataset/QH_160/test/000_quang_hai_0.png"]
            image = facenet.load_data(test_path, False, False,160)
            test_emb = np.zeros((1, embedding_size))
            feed_dict = {images_placeholder: image, phase_train_placeholder: False}
            test_emb[0, :] = sess.run(embeddings, feed_dict=feed_dict)

            # Embed test image
            print("Emb db image")
            data_dir = "/home/thanhnn/dataset/QH_160/Extracted/quang_hai"
            for image_name in os.listdir(data_dir):
                image_path = os.path.join(data_dir, image_name)
                test_path = [image_path]
                image = facenet.load_data(test_path, False, False, 160)
                db_emb = np.zeros((1, embedding_size))
                feed_dict_2 = {images_placeholder: image, phase_train_placeholder: False}
                db_emb[0, :] = sess.run(embeddings, feed_dict=feed_dict_2)

                # Compare distance
                dist = euclidean_distances(test_emb, db_emb)
                dist = dist /np.max(dist)

                # Compare threshold
                threshold = 0.5
                if dist[0][0] < threshold:
                    print("Den")
                    pass
                else:
                    print("Den")

if __name__ == '__main__':
    main()
