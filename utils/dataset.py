# Copyright 2017 Google Inc. All Rights Reserved.
#
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
# ==============================================================================

"""For loading data into NMT cnn_models."""
from __future__ import print_function

import collections
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
    pass


def get_number_of_frames(src):
    src = src.numpy()  # os.listdir(path), path cannot be tensor.
    return np.int32(len([f for f in listdir(src) if isfile(join(src, f))]))


def read_video(src, source_reverse):
    src = src.numpy()
    images = sorted([f for f in listdir(src) if isfile(join(src, f))])
    # video = np.zeros((len(images), 227, 227, 3)).astype(np.float32)
    video = np.zeros((len(images), 112, 112, 3)).astype(np.float32)

    # Cihan_CR: Harcoded Path, Need to Change This
    mean_image = np.load('/home/panxie/Documents/sign-language/nslt/Mean/'
                         'FulFrame_Mean_Image_227x227.npy').astype(np.float32)[..., ::-1]

    # for each image
    for i in range(0, len(images)):
        img_path = str(src + images[i], encoding="utf-8")
        # video[i, :, :, :] = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32) - mean_image
        video[i, :, :, :] = cv2.resize(cv2.imread(img_path), (112, 112)).astype(np.float32)

    if source_reverse:
        video = np.flip(video, axis=0)

    return video


# 定义bucket的最小和最大边界，以及
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.4


def _batch_examples(dataset, batch_size, src_max_length):
    buckets_min, buckets_max = _create_min_max_boundaries(src_max_length)

    bucket_batch_sizes = [batch_size // x for x in buckets_max]
    bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

    def example_to_bucket_id(example_input, example_target, _1, _2, _3):
        """Return int64 bucket id for this example, calculated based on length."""
        seq_length = _get_example_length((example_input, example_target))

        # TODO: investigate whether removing code branching improves performance.
        conditions_c = tf.logical_and(
            tf.less_equal(buckets_min, seq_length),  # Tensor("LessEqual_1:0", shape=(24,), dtype=bool)
            tf.less(seq_length, buckets_max))
        bucket_id = tf.reduce_min(tf.where(conditions_c))
        return bucket_id  # Tensor("Min:0", shape=(), dtype=int64)

    def window_size_fn(bucket_id):
        """Return number of examples to be grouped when given a bucket id."""
        return bucket_batch_sizes[bucket_id]

    def batching_fn(bucket_id, grouped_dataset):
        """Batch and add padding to a dataset of elements with similar lengths."""
        bucket_batch_size = window_size_fn(bucket_id)
        # Batch the dataset and add padding so that all input sequences in the
        # examples have the same length, and all target sequences have the same
        # lengths as well. Resulting lengths of inputs and targets can differ.
        return grouped_dataset.padded_batch(bucket_batch_size, ([None, None, None, None],
                                                                [None], [None], [], []))

    return dataset.apply(tf.data.experimental.group_by_window(
        key_func=example_to_bucket_id,
        reduce_func=batching_fn,
        window_size=None,
        window_size_func=window_size_fn))


def _get_example_length(example):
    # print(tf.shape(example[0]), tf.shape(example[1]))
    # exit()
    """Returns the maximum length between the example inputs and targets."""
    length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
    return length


def _create_min_max_boundaries(
        max_length, min_boundary=_MIN_BOUNDARY, boundary_scale=_BOUNDARY_SCALE):
    """Create min and max boundary lists up to max_length.
    For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
    returned values will be:
      buckets_min = [0, 4, 8, 16, 24]
      buckets_max = [4, 8, 16, 24, 25]
    Args:
      max_length: The maximum length of example in dataset.
      min_boundary: Minimum length in boundary.
      boundary_scale: Amount to scale consecutive boundaries in the list.
    Returns:
      min and max boundary lists
    """
    # Create bucket boundaries list by scaling the previous boundary or adding 1
    # (to ensure increasing boundary sizes).
    bucket_boundaries = []
    x = min_boundary
    while x < max_length:
        bucket_boundaries.append(x)
        x = max(x + 1, int(x * boundary_scale))

    # Create min and max boundary lists from the initial list.
    buckets_min = [0] + bucket_boundaries
    buckets_max = bucket_boundaries + [max_length + 1]
    return buckets_min, buckets_max


def get_iterator(src_dataset,
                 tgt_dataset,
                 tgt_vocab_table,
                 sos,
                 eos,
                 source_reverse,
                 random_seed,
                 batch_size=301,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=None,
                 skip_count=None):
    # Cihan_CR: Hard Codded - Need to Change this
    # if not output_buffer_size:
    #     output_buffer_size = 10  # batch_size * 1000

    output_buffer_size = 10

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    # Concat Datasets
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    # Skip Data
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    # Shuffle Samples: You must do it as early as possible
    src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size * 1000, random_seed)

    # Get number of frames from videos
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt, tf.py_function(get_number_of_frames, [src], tf.int32)),
        num_parallel_calls=num_parallel_calls)  # src_path, tgt_string, src_len

    # Split Translation into Tokens
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tf.strings.split([tgt]).values, src_len),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_tokens, src_len

    # Sequence Length Checks
    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt, src_len: tf.logical_and(src_len > 0, tf.size(tgt) > 0))
    src_tgt_dataset = src_tgt_dataset.filter(
        lambda src, tgt, src_len: tf.logical_and(src_len < src_max_len, tf.size(tgt) < tgt_max_len))

    # Convert Tokens to IDs
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tf.cast(tgt_vocab_table.lookup(tgt), tf.int32), src_len),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_ids, src_len

    # Create Input and Output for Target
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src,
                                           tf.concat(([tgt_sos_id], tgt), 0),
                                           tf.concat((tgt, [tgt_eos_id]), 0),
                                           src_len),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_in_ids, tgt_out_ids, src_len

    # Get Target Sequence Length
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, src_len:
                                          (src, tgt_in, tgt_out, src_len, tf.size(tgt_in)),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_in_ids, tgt_out_ids, src_len, tgt_len

    # Read video, transfer video path to tensor
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt_in, tgt_out, src_len, tgt_len:
                                          (tf.py_function(read_video, [src, source_reverse], tf.float32),
                                           tgt_in, tgt_out, src_len, tgt_len),
                                          num_parallel_calls=num_parallel_calls)  # src_video, tgt_in_ids, tgt_out_ids, src_len, tgt_len

    # src_tgt_dataset = src_tgt_dataset.padded_batch(batch_size=2,
    #                                                padded_shapes=([None, None, None, None],
    #                                                               [None],[None], [], []))

    src_tgt_dataset = _batch_examples(src_tgt_dataset, batch_size=batch_size, src_max_length=300)

    src_tgt_dataset = src_tgt_dataset.repeat(count=1)

    # Create Initializer
    return src_tgt_dataset


def get_infer_iterator(src_dataset, source_reverse):
    # Get number of Frames
    src_dataset = src_dataset.map(lambda src: (src, tf.py_function(get_number_of_frames, [src], tf.int32)))
    # for data in src_dataset.take(1):
    #     print(data)
    #     exit()
    # Filter Out Samples
    # src_dataset = src_dataset.filter(lambda src, src_len: tf.logical_and(src_len > 0, src_len < src_max_len))

    src_dataset = src_dataset.map(lambda src, src_len:
                                  (tf.py_function(read_video, [src, source_reverse], tf.float32),
                                   tf.reshape(src_len, [1])))
    src_dataset = src_dataset.map(lambda src, src_len: (tf.expand_dims(src, axis=0), src_len))

    src_dataset = src_dataset.repeat(1)
    return src_dataset


def get_infer_iterator_2(src_dataset, tgt_dataset, tgt_vocab_table, source_reverse, num_parallel_calls=None):
    # Get number of Frames
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
    # Get number of frames from videos
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt, tf.py_function(get_number_of_frames, [src], tf.int32)),
        num_parallel_calls=num_parallel_calls)  # src_path, tgt_string, src_len
    # Split Translation into Tokens
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tf.strings.split([tgt]).values, src_len),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_tokens, src_len
    # Convert Tokens to IDs
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tf.cast(tgt_vocab_table.lookup(tgt), tf.int32), src_len),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_ids, src_len

    # Get Target Sequence Length
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len:
                                          (src, tgt, src_len, tf.size(tgt)),
                                          num_parallel_calls=num_parallel_calls)  # src_path, tgt_in_ids, tgt_out_ids, src_len, tgt_len

    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len, tgt_len:
                                  (tf.py_function(read_video, [src, source_reverse], tf.float32), tgt,
                                   tf.reshape(src_len, [1]), tf.reshape(tgt_len, [1])))
    src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, src_len, tgt_len: (tf.expand_dims(src, axis=0), tf.expand_dims(tgt, axis=0), src_len, tgt_len))

    src_tgt_dataset = src_tgt_dataset.repeat(1)
    return src_tgt_dataset


def get_train_dataset(src_file, tgt_file, tgt_vocab_table, batch_size=301):
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)

    iterator = get_iterator(src_dataset,
                            tgt_dataset,
                            tgt_vocab_table,
                            sos="<s>",
                            eos="</s>",
                            batch_size=batch_size,
                            source_reverse=True,
                            random_seed=285,
                            src_max_len=300,
                            tgt_max_len=50,
                            skip_count=None)
    return iterator

def get_infer_dataset(src_file, tgt_file, tgt_vocab_table, source_reverse=False):
    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)

    iterator = get_infer_iterator_2(src_dataset, tgt_dataset, tgt_vocab_table, source_reverse=source_reverse)
    return iterator

if __name__ == "__main__":
    from utils.vocab_utils import create_tgt_vocab_table

    print(tf.__version__)
    import os

    # print(os.getcwd())
    base_path = "/home/panxie/Documents/sign-language/nslt/Data"
    src_file = base_path + "/phoenix2014T.test.sign"
    tgt_file = base_path + "/phoenix2014T.test.de"
    tgt_vocab_table = create_tgt_vocab_table(base_path + "/phoenix2014T.vocab.de")
    # dataset = get_train_dataset(src_file, tgt_file, tgt_vocab_table)
    # cnt = 0
    # for data in dataset.take(-1):
    #     cnt += 1
    #     print(data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape)
    # print(cnt)
    dataset = get_infer_dataset(src_file, tgt_file, tgt_vocab_table)
    cnt = 0
    for data in dataset.take(-1):
        cnt += 1
        print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
        print(cnt)