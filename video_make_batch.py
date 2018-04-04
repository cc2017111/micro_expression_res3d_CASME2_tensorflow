import tensorflow as tf
import video_opts
import numpy as np


def read_and_decode(filename):
    reader = tf.TFRecordReader()
    args = video_opts.parse_opts()
    filename_queue = tf.train.string_input_producer([filename], shuffle=False, num_epochs=args.epoch, capacity=256)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={'label': tf.FixedLenFeature([], tf.int64),
                  'video_raw': tf.FixedLenFeature([], tf.string)})
    video = tf.decode_raw(features['video_raw'], tf.uint8)
    channel = 3 if args.color else 1
    video = tf.reshape(video, [args.depth, args.img_size, args.img_size, channel])
    video = tf.cast(video, tf.float32)
    labels = tf.cast(features['label'], tf.int32)
    return video, labels


def get_batch(filename, batch_size):
    # To change data type
    video, label = read_and_decode(filename)

    # To create batch
    video_batch, label_batch = tf.train.shuffle_batch([video, label],
                                                      batch_size=batch_size,
                                                      num_threads=32,
                                                      capacity=256,
                                                      min_after_dequeue=1)
    # To reshape label
    label_batch = tf.reshape(label_batch, [batch_size])
    return video_batch, label_batch

