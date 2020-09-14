#!/usr/bin/env python3
#
# Practical Applications of Deep Learning: Classifying the most common categories of plain radiographs in a PACS using a neural network (NNCPR)
# Thomas Dratsch; Michael Korenkov; David Zopfs; Sebastian Brodehl; Bettina Baessler; Daniel Giese; Sebastian Brinkmann; David Maintz; Daniel Pinto dos Santos
# European Radiology
# https://github.com/healthcAIr/NNCPR
#
# Copyright 2020 The NNCPR Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tensorflow as tf


def load_graph(model_file: str):
    """Load graph definition from file."""
    g = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with g.as_default():
        tf.import_graph_def(graph_def)

    return g


def read_image_and_preprocess(file_name: str,
                              input_height: int = 299, input_width: int = 299,
                              input_mean: int = 0, input_std: int = 255):
    """Read image from disk and do preprocessing (resizing & normalization)."""
    file_reader = tf.io.read_file(file_name, "file_reader")
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    with tf.compat.v1.Session() as s:
        result = s.run(normalized)

    return result


def load_labels(label_file: str):
    """Load list of label names from ASCII file."""
    return [l.strip() for l in tf.io.gfile.GFile(label_file).readlines()]


if __name__ == "__main__":
    import os
    import argparse
    import time
    import numpy as np
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument("images", help="image(s) to be processed", nargs='+')
    parser.add_argument("--graph", help="pretrained graph/model to be executed", default="network.pb")
    parser.add_argument("--labels", help="ASCII file containing labels", default="classes.txt")
    parser.add_argument("--input_height", type=int, help="input height", default=224)
    parser.add_argument("--input_width", type=int, help="input width", default=224)
    parser.add_argument("--input_mean", type=int, help="input mean", default=128)
    parser.add_argument("--input_std", type=int, help="input std", default=128)
    parser.add_argument("--input_layer", help="name of input layer in graph", default="input")
    parser.add_argument("--output_layer", help="name of output layer in graph", default="final_result")
    args = parser.parse_args()

    # load pretrained graph definition
    graph = load_graph(args.graph)
    # define input and output tensors
    input_operation = graph.get_operation_by_name("import/" + args.input_layer)
    output_operation = graph.get_operation_by_name("import/" + args.output_layer)

    # load class labels from file
    labels = load_labels(args.labels)

    # iterate over all input images to classifiy
    for image_filepath in args.images:
        # load image and do some preprocessing
        t = read_image_and_preprocess(image_filepath,
                                      input_height=args.input_height,
                                      input_width=args.input_width,
                                      input_mean=args.input_mean,
                                      input_std=args.input_std)
        # run image through the pretrained network
        with tf.compat.v1.Session(graph=graph) as sess:
            start_time = time.perf_counter()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            inference_time = time.perf_counter() - start_time

        # look up predicted label
        results = np.squeeze(results)
        top_5 = results.argsort()[-5:][::-1]
        print("{} \t {} \t {:0.5f} \t {:.2f}s" .format(
            image_filepath, labels[top_5[0]], results[top_5[0]], inference_time)
        )
