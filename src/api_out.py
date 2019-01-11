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
import facenet_temp as facenet
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
    image = cv2.imread(img_path)
    # print(image.shape)

    for top, left, bottom, right in predictions:
        cv2.rectangle(image, (top, left), (bottom, right), (0, 255, 0), 3)
    # print(save_path)
    cv2.imwrite(save_path, image)


def main(model_path, data_dir, base_data_dir, output_data_dir, test_path, bb_path, result_path, final_path):
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=666)

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # TODO
            # Embed test image
            print("Emb test image")
            test_path = [test_path]
            image = facenet.load_data(test_path, False, False, 160)
            test_emb = np.zeros((1, embedding_size))
            feed_dict = {images_placeholder: image,
                         phase_train_placeholder: False}
            test_emb[0, :] = sess.run(embeddings, feed_dict=feed_dict)

            # Embed test image
            print("Emb db image")
            count = 0
            result_name = []
            with open(final_path, "w") as ffinal:
                ffinal.write("name, top, left, bottom, right\n")
                listdir = sorted(os.listdir(data_dir))
                for image_name in listdir:
                    image_path = os.path.join(data_dir, image_name)
                    test_path = [image_path]
                    image = facenet.load_data(test_path, False, False, 160)
                    db_emb = np.zeros((1, embedding_size))
                    feed_dict_2 = {images_placeholder: image,
                                   phase_train_placeholder: False}
                    db_emb[0, :] = sess.run(embeddings, feed_dict=feed_dict_2)

                    # Compare distance
                    dist = euclidean_distances(test_emb, db_emb)
                    # dist = dist / np.max(dist)

                    # Compare threshold
                    threshold = 1.0
                    if dist[0][0] < threshold:
                        count += 1
                        line = None
                        top, right, bottom, left = 0, 0, 0, 0
                        with open(bb_path, "r") as f:
                            lines = f.readlines()
                            for line in lines:
                                if image_path in line:
                                    temp = line.strip().split(" ")
                                    top, left, bottom, right = int(temp[1]), int(
                                        temp[2]), int(temp[3]), int(temp[4])
                                    break

                        predictions = [(top, left, bottom, right)]

                        base_image_component = image_name.split("_")
                        base_image_component.pop(-1)
                        base_image_name = "_".join(
                            base_image_component) + ".jpg"
                        if base_image_name not in result_name:
                            result_name.append(base_image_name)

                        # base_image_path = os.path.join(base_data_dir, base_image_name)
                        output_image_path = os.path.join(
                            output_data_dir, base_image_name)
                        if os.path.isfile(output_image_path):
                            base_image_path = output_image_path
                        else:
                            base_image_path = os.path.join(
                                base_data_dir, base_image_name)

                        show_prediction_labels_on_image(
                            base_image_path, predictions, output_image_path)

                        ffinal.write(base_image_name + "," + str(top) + "," +
                                     str(left) + "," + str(bottom) + "," + str(right) + "\n")

            print("Total: %d" % count)

            with open(result_path, "w") as f:
                f.write("image, isQH\n")
                for img_name in os.listdir(base_data_dir):
                    if img_name in result_name:
                        f.write(img_name + ", true\n")
                    else:
                        f.write(img_name + ", false\n")
