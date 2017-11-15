#coding: utf-8
import tensorflow as tf
import numpy as np
import utils
import config
import os, glob
import scipy.misc
import cv2
from argparse import ArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

slim = tf.contrib.slim


def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.model_zoo)
    parser.add_argument('--model', help=models_str, required=True)
    parser.add_argument('--name', help='default: name=model')
    parser.add_argument('--sample_size','-N',help='# of samples.It should be a square number. (default: 16)',default=16,type=int)

    return parser

def pre_precess_LR(im, crop_size):
    output_height, output_width = crop_size
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("image is small")

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    im = im[offset_h:offset_h+output_height, offset_w:offset_w+output_width, :]
    LR = scipy.misc.imresize(im,[8,8])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 8, 8, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])


    return im_LR,im_HR,LR,HR
#prec msc
def pre_precess_msc(im, crop_size):

    im = im[152-64:152+64, 122-64:122+64, :]
    LR = scipy.misc.imresize(im,[16,16])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])


    return im_LR,im_HR,LR,HR
def pre_precess_lan(im, crop_size):

    #im = im[152-64:152+64, 122-64:122+64, :]
    LR = scipy.misc.imresize(im,[16,16])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])


    return im_LR,im_HR,LR,HR
def pre_precess_lan_8(im, crop_size):

    #im = im[152-64:152+64, 122-64:122+64, :]
    LR = scipy.misc.imresize(im,[8,8])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 8, 8, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])


    return im_LR,im_HR,LR,HR
def sample_z(shape):
    return np.random.normal(size=shape)


def get_all_checkpoints(ckpt_dir, force=False):
    '''
    When the learning is interrupted and resumed, all checkpoints can not be fetched with get_checkpoint_state
    (The checkpoint state is rewritten from the point of resume).
    This function fetch all checkpoints forcely when arguments force=True.
    '''

    if force:
        ckpts = os.listdir(ckpt_dir) # get all fns
        ckpts = map(lambda p: os.path.splitext(p)[0], ckpts) # del ext
        ckpts = set(ckpts) # unique
        ckpts = filter(lambda x: x.split('-')[-1].isdigit(), ckpts) # filter non-ckpt
        ckpts = sorted(ckpts, key=lambda x: int(x.split('-')[-1])) # sort
        ckpts = map(lambda x: os.path.join(ckpt_dir, x), ckpts) # fn => path
    else:
        ckpts = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths

    return ckpts


def eval(model, name, sample_shape=[1,1], load_all_ckpt=True):
    if name == None:
        name = model.name
    dir_name = 'eval/' + name
    if tf.gfile.Exists(dir_name):
        tf.gfile.DeleteRecursively(dir_name)
    tf.gfile.MakeDirs(dir_name)
    dir_name_sr = dir_name + '/sr'
    dir_name_lr = dir_name + '/lr'
    dir_name_hr = dir_name + '/hr'
    tf.gfile.MakeDirs(dir_name_sr)
    tf.gfile.MakeDirs(dir_name_lr)
    tf.gfile.MakeDirs(dir_name_hr)

    # training=False => generator only
    restorer = tf.train.Saver(slim.get_model_variables())

    config = tf.ConfigProto()
    # best_gpu = utils.get_best_gpu()
    config.gpu_options.allow_growth = True # Works same as CUDA_VISIBLE_DEVICES!
    with tf.Session(config=config) as sess:
        ckpts = get_all_checkpoints('./checkpoints/' + name, force=load_all_ckpt)
        restorer.restore(sess, ckpts[-1])
        size = sample_shape[0] * sample_shape[1]
        #fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/msc_test','r')
        fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/celeA_List/test_lan_100','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/lan_paper_part','r')
        # path_image = '/home/safe_data_dir_2/guoyu/png_lanzhou_crop/'
        # path_image = "/home/tmp_data_dir/zhaoyu/CelebA/img_align_celeba/"
        #path_image = '/home/xujinchang/share/project/GAN/tf_WGAN_GP/m.0bh0sn/'
        path_image = "/home/xujinchang/100_128face/"
        image_list  = []
        for line in fp.readlines():
            image_list.append(line.strip().split(' ')[0])
        count = 0
        for image in image_list:
            count += 1
            print("count: {}".format(count))
            im = scipy.misc.imread(path_image+image, mode='RGB')
            z_, gt, LR, HR = pre_precess_lan_8(im,[128,128])
            fn_lr = "lr_"+image
            fn_hr = "hr_"+image
            scipy.misc.imsave(os.path.join(dir_name_lr, fn_lr),LR)
            scipy.misc.imsave(os.path.join(dir_name_hr, fn_hr),HR)
            z_ = z_ / 127.5 - 1.0

            fake_samples = sess.run(model.fake_sample, {model.z: z_})
            fake_samples = (fake_samples + 1.) * 127.5
            merged_samples = utils.merge(fake_samples, size=sample_shape)
            fn = "sr_"+image
            scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)
        # for v in ckpts:
        #     count += 1
        #     # if count < 50: continue
        #     print("Evaluating {} ...".format(v))
        #     restorer.restore(sess, v)
        #     global_step = int(v.split('/')[-1].split('-')[-1])

        #     fake_samples = sess.run(model.fake_sample, {model.z: z_})

        #     # inverse transform: [-1, 1] => [0, 1]
        #     fake_samples = (fake_samples + 1.) / 2.
        #     merged_samples = utils.merge(fake_samples, size=sample_shape)
        #     fn = "{:0>5d}.png".format(global_step)
        #     scipy.misc.imsave(os.path.join(dir_name, fn), merged_samples)

'''
You can create a gif movie through imagemagick on the commandline:
$ convert -delay 20 eval/* movie.gif
'''
# def to_gif(dir_name='eval'):
#     images = []
#     for path in glob.glob(os.path.join(dir_name, '*.png')):
#         im = scipy.misc.imread(path)
#         images.append(im)

#     # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
#     imageio.mimsave('movie.gif', images, duration=0.2)

if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model = FLAGS.model.upper()
    if FLAGS.name is None:
        FLAGS.name = FLAGS.model.lower()
    config.pprint_args(FLAGS)

    N = FLAGS.sample_size**0.5
    assert N == int(N), 'sample size should be a square number'
    model = config.get_model(FLAGS.model, FLAGS.name, training=False)
    eval(model, name=FLAGS.name, sample_shape=[1,1], load_all_ckpt=True)
