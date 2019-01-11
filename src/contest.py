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


def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)
            dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths) > 0,
                       'There must be at least one image for each class in the dataset')

            paths, labels, pos = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            print("Index QH: {}".format(labels[pos]))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')


            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(
                math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            
            control_array = np.ones_like(np.expand_dims(labels, 1), np.int32)*facenet.FIXED_STANDARDIZATION
            print(control_array.shape)

            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(
                    paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images,
                             phase_train_placeholder: False,
                             control_placeholder: control_array}
                emb_array[start_index:end_index, :] = sess.run(
                    embeddings, feed_dict=feed_dict)
            

            # with open("embeddings.pkl", "wb") as f:
            #     pickle.dump(emb_array, f)
            # with open("labels.pkl", "wb") as fl:
            #     pickle.dump(labels, fl)

            # with open("embeddings.pkl", "rb") as f:
            #     emb_array = pickle.load(f)
            # with open("labels.pkl", "rb") as fl:
            #     labels = pickle.load(fl)

            # TODO
            # Embed test image
            print("Test with image")
            test_path = ["/home/thanhnn/dataset/QH_160/test/000_quang_hai_0.png"]
            image = facenet.load_data(test_path, False, False, args.image_size)
            test_emb = np.zeros((1, embedding_size))
            
            control_array = np.array([[1]], np.int32)*facenet.FIXED_STANDARDIZATION
            print(control_array.shape)
            
            feed_dict_2 = {images_placeholder: image, phase_train_placeholder: False, control_placeholder: control_array}
            test_emb[0, :] = sess.run(embeddings, feed_dict=feed_dict_2)

            # Compare distance
            # dist = np.linalg.norm(emb_array - test_emb)
            dist = euclidean_distances(test_emb, emb_array)
            dist = dist /np.max(dist)
            # print(np.min(dist))
            # print(np.max(dist))

            ## TODO
            # Calculate tpr, fpr, acc
            # actual_issame = labels == 5750
            # tpr, fpr, acc = facenet.calculate_accuracy(0.7, dist, actual_issame)
            # print("tpr: {}, fpr: {}, acc: {}".format(tpr, fpr, acc))
            thresholds = np.arange(0.3, 0.6, 0.01)
            best_f1 = 0
            best_threshold = 0
            for threshold in thresholds:
                # threshold = 0.5
                index = [i for i in range(emb_array.shape[0]) if dist[0][i] < threshold]
                label = [dist[0][idx] for idx in index]

                tp = len([item for item in index if labels[item] == 5750])
                actual = 92
                pred = len(index)
                precision = float(tp) / float(pred)
                recall = float(tp) / float(actual)
                f1 = 2.0 / (1/precision + 1/recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thres = threshold
                print("Threshold: {0:.2f}".format(threshold))
                # print(" + true: %d" % len([item for item in index if labels[item] == 5750]))
                # print(" + pred: %d" % len(index))
                print("F1 score: {0:.2f}".format(f1))


            print("Best threshold: {0:.2f}, best f1: {0:.2f}".format(best_threshold, best_f1))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
