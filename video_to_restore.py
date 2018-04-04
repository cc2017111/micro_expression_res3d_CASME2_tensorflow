import os
import tensorflow as tf
from tqdm import tqdm
import video_opts
import video_to_3d_tensor
import numpy as np
import video_classes


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value={value}))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value={value}))


def video_to_tfrecord(video_dir, vid3d, nclass, result_dir, mapfile, color=False, skip=True):

    labels = []
    labellist = []
    files = os.listdir(video_dir)
    pbar = tqdm(total=len(files))
    writer = tf.python_io.TFRecordWriter(result_dir)

    for dictionary in files:
        filepath = os.path.join(video_dir, dictionary)
        pbar.update(1)
        index = 0
        for filename in os.listdir(filepath):
            name = os.path.join(filepath, filename)
            print(name)
            label = dictionary
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            labels.append(label)
            video_raw = vid3d.Video_3D(name, color=color, skip=skip)
            for num, class_name in video_classes.classes.items():
                if label == class_name:
                    index = num
                    break

            video_raw_bytes = video_raw.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'video_raw': _bytes_feature(video_raw_bytes)}))

            writer.write(example.SerializeToString())

    writer.close()
    pbar.close()
    with open(os.path.join(mapfile, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

args = video_opts.parse_opts()
img_size, frames = args.img_size, args.depth
channel = 3 if args.color else 1

vid3d = video_to_3d_tensor.Video_to_3dtensor(args.img_size, frames)

video_to_tfrecord(args.videos, vid3d, args.nclasses, args.output, args.mapfile, args.color, args.skip)
print('Saved dataset as tfrecords.')
