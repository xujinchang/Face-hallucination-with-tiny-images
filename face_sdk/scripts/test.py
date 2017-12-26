#-*- coding:utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
import scipy.misc


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')

def load_graph(frozen_graph_filename):
    #  parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #  load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="nn_new.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    #op.values() gives you a list of tensors it produces
    #op.name gives you the name
    #for op in graph.get_operations():
    #    print(op.name,op.values())
    #    break
    x = graph.get_tensor_by_name('prefix/WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/Placeholder_1:0')
    y = graph.get_tensor_by_name('prefix/WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/generator/Conv/Tanh:0')
    image = 'input.jpg'
    im = scipy.misc.imread(image, mode='RGB')
    LR = scipy.misc.imresize(im,[16,16])
    z_ = np.reshape(LR, [-1, 16, 16, 3])

    z_ = z_ / 127.5 - 1.0

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={x: z_})
        y_out = (y_out + 1.0) * 127.5
        merged_samples = merge(y_out, [1, 1])
        scipy.misc.imsave('output.jpg', merged_samples)
    print ("finish")
