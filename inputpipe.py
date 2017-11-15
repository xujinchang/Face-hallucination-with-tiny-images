# coding: utf-8
import tensorflow as tf


# def read_parse_preproc(filename_queue):
#     ''' read, parse, and preproc single example. '''
#     with tf.variable_scope('read_parse_preproc'):
#         reader = tf.TFRecordReader()
#         key, records = reader.read(filename_queue)

#         # parse records
#         features = tf.parse_single_example(
#             records,
#             features={
#                 "image_LR": tf.FixedLenFeature([], tf.string),
#                 "image_HR": tf.FixedLenFeature([], tf.string)
#             }
#         )

#         image_HR = tf.decode_raw(features["image_HR"], tf.uint8)
#         image_HR = tf.reshape(image_HR, [64, 64, 3]) # The image_shape must be explicitly specified
#         image_HR = tf.image.resize_images(image_HR, [64, 64])
#         # image_HR = tf.image.resize_images(image_HR, [16, 16])
#         # image_HR = tf.image.random_brightness(image_HR, 0.5)
#         image_HR = tf.cast(image_HR, tf.float32)
#         image_HR = image_HR / 127.5 - 1.0 # preproc - normalize


#         image_LR = tf.decode_raw(features["image_LR"], tf.uint8)
#         image_LR = tf.reshape(image_LR, [16, 16, 3]) # The image_shape must be explicitly specified
#         image_LR = tf.image.resize_images(image_LR, [16, 16])
#         # image_LR = tf.image.random_brightness(image_LR, 0.5)
#         image_LR = tf.cast(image_LR, tf.float32)

#         image_LR = image_LR / 127.5 - 1.0 # preproc - normalize
#         return [image_HR, image_LR]
def read_parse_preproc(filename_queue):
    ''' read, parse, and preproc single example. '''
    with tf.variable_scope('read_parse_preproc'):
        reader = tf.TFRecordReader()
        key, records = reader.read(filename_queue)

        # parse records
        features = tf.parse_single_example(
            records,
            features={
                "image_LR": tf.FixedLenFeature([], tf.string),
                "image_HR": tf.FixedLenFeature([], tf.string)
            }
        )

        image_HR = tf.decode_raw(features["image_HR"], tf.uint8)
        image_HR = tf.reshape(image_HR, [64, 64, 3]) # The image_shape must be explicitly specified
        image_HR = tf.image.resize_images(image_HR, [64, 64])
        # image_HR = tf.image.resize_images(image_HR, [16, 16])
        # image_HR = tf.image.random_brightness(image_HR, 0.5)
        image_HR = tf.cast(image_HR, tf.float32)
        image_HR = image_HR / 127.5 - 1.0 # preproc - normalize


        image_LR = tf.decode_raw(features["image_LR"], tf.uint8)
        image_LR = tf.reshape(image_LR, [8, 8, 3]) # The image_shape must be explicitly specified
        image_LR = tf.image.resize_images(image_LR, [8, 8])
        # image_LR = tf.image.random_brightness(image_LR, 0.5)
        image_LR = tf.cast(image_LR, tf.float32)

        image_LR = image_LR / 127.5 - 1.0 # preproc - normalize
        return [image_HR, image_LR]

def read_parse_preproc_big(filename_queue):
    ''' read, parse, and preproc single example. '''
    with tf.variable_scope('read_parse_preproc_big'):
        reader = tf.TFRecordReader()
        key, records = reader.read(filename_queue)

        # parse records
        features = tf.parse_single_example(
            records,
            features={
                "image_LR": tf.FixedLenFeature([], tf.string),
                "image_HR": tf.FixedLenFeature([], tf.string)
            }
        )

        image_HR = tf.decode_raw(features["image_HR"], tf.uint8)
        image_HR = tf.reshape(image_HR, [64, 64, 3]) # The image_shape must be explicitly specified
        image_HR = tf.image.resize_images(image_HR, [64, 64])
        image_HR = tf.cast(image_HR, tf.float32)
        image_HR = image_HR / 127.5 - 1.0 # preproc - normalize


        image_LR = tf.decode_raw(features["image_LR"], tf.uint8)
        image_LR = tf.reshape(image_LR, [16, 16, 3]) # The image_shape must be explicitly specified
        image_LR = tf.image.resize_images(image_LR, [16, 16])
        image_LR = tf.cast(image_LR, tf.float32)
        image_LR = image_LR / 127.5 - 1.0 # preproc - normalize
        return [image_HR, image_LR]

# https://www.tensorflow.org/programmers_guide/reading_data
def get_batch(tfrecords_list, batch_size, shuffle=False, num_threads=1, min_after_dequeue=None, num_epochs=None):
    name = "batch" if not shuffle else "shuffle_batch"
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=shuffle, num_epochs=num_epochs)
        # change 64x64 to 128x128

        data_point = read_parse_preproc(filename_queue)

        if min_after_dequeue is None:
            min_after_dequeue = batch_size * 10
        capacity = min_after_dequeue + 3*batch_size
        if shuffle:
            batch_HR, batch_LR = tf.train.shuffle_batch(data_point, batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue, num_threads=num_threads, allow_smaller_final_batch=True)
        else:
            batch_HR, batch_LR = tf.train.batch(data_point, batch_size, capacity=capacity, num_threads=num_threads,
                allow_smaller_final_batch=True)

        return batch_HR, batch_LR


def get_batch_join(tfrecords_list, batch_size, shuffle=False, num_threads=1, min_after_dequeue=None, num_epochs=None):
    name = "batch_join" if not shuffle else "shuffle_batch_join"
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=shuffle, num_epochs=num_epochs)
        example_list = [read_parse_preproc(filename_queue) for _ in range(num_threads)]

        if min_after_dequeue is None:
            min_after_dequeue = batch_size * 10
        capacity = min_after_dequeue + 3*batch_size
        if shuffle:
            batch = tf.train.shuffle_batch_join(tensors_list=example_list, batch_size=batch_size, capacity=capacity,
                                                min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
        else:
            batch = tf.train.batch_join(example_list, batch_size, capacity=capacity, allow_smaller_final_batch=True)

        return batch


# interfaces
def shuffle_batch_join(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch_join(tfrecords_list, batch_size, shuffle=True, num_threads=num_threads,
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)

def batch_join(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch_join(tfrecords_list, batch_size, shuffle=False, num_threads=num_threads,
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)

def shuffle_batch(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch(tfrecords_list, batch_size, shuffle=True, num_threads=num_threads,
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)

def batch(tfrecords_list, batch_size, num_threads, num_epochs, min_after_dequeue=None):
    return get_batch(tfrecords_list, batch_size, shuffle=False, num_threads=num_threads,
        num_epochs=num_epochs, min_after_dequeue=min_after_dequeue)
