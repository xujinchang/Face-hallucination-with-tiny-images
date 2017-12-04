#coding: utf-8
import tensorflow as tf
import numpy as np
import utils
import config
import os, glob
import scipy.misc
import cv2
from argparse import ArgumentParser
from face_warp import *
from scipy import ndimage
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
def sample_z(shape):
    return np.random.uniform(-0.1,0.1,size=shape)

    return im_LR,im_HR,LR,HR
def pre_precess_lan_20(im, mean_value, std_value):

    # 105.687841643 2660.10447792 std = 51.57, mean = 105.68
    #im = im[152-64:152+64, 122-64:122+64, :]
    # mean_value = 105.69 - factor
    # std_value = 60 - factor
    # mean_face = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face.jpg',mode='RGB')
    # print "mean: std: ", np.mean(mean_face), np.std(mean_face)
    LR = scipy.misc.imresize(im,[16,16])
    # LR = LR + 0.2*mean_face

    LR = mean_value + std_value * (LR - np.mean(LR)) / np.sqrt(np.var(LR))
    # LR = (LR - mean_LR) * 1.5 + 1.2*mean_LR
    LR[LR > 255] = 255
    LR[LR < 0] = 0
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])
    return im_LR,im_HR,LR,HR

def pre_precess_lan_5(im, factor1, factor2):

    # 105.687841643 2660.10447792 std = 51.57, mean = 105.68
    #im = im[152-64:152+64, 122-64:122+64, :]
    # mean_value = 105.69 - factor
    # std_value = 60 - factor
    # mean_face = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face.jpg',mode='RGB')
    # print "mean: std: ", np.mean(mean_face), np.std(mean_face)
    LR = scipy.misc.imresize(im,[16,16])
    # LR = LR + 0.2*mean_face
    LR = (LR - mean_LR) * factor1 + factor2*mean_LR
    LR[LR > 255] = 255
    LR[LR < 0] = 0
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])
    return im_LR,im_HR,LR,HR

def pre_precess_mean(im, mean_face, factor):

    # 105.687841643 2660.10447792 std = 51.57, mean = 105.68
    #im = im[152-64:152+64, 122-64:122+64, :]
    # mean_value = 105.69 - factor
    # std_value = 60 - factor
    # 
    # print "mean: std: ", np.mean(mean_face), np.std(mean_face)
    LR = scipy.misc.imresize(im,[16,16])
    LR = (LR + factor*mean_face) / (1+factor)
    # LR = (LR - mean_LR) * factor1 + factor2*mean_LR
    LR[LR > 255] = 255
    LR[LR < 0] = 0
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
    im_HR = np.reshape(HR, [-1, 64, 64, 3])
    return im_LR,im_HR,LR,HR

def pre_precess_lfw_8(im, landmarks):

    #im = im[152-64:152+64, 122-64:122+64, :]
    im = face_warp_main(im, landmarks, "origin")
    LR = scipy.misc.imresize(im,[16,16])
    HR = scipy.misc.imresize(im,[64,64])
    # scipy.misc.imsave('119956_lr.jpg', im_LR)
    # scipy.misc.imsave('119956_hr.jpg', im_HR)
    im_LR = np.reshape(LR, [-1, 16, 16, 3])
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
        ckpts = get_all_checkpoints('/home/xujinchang/share/project/GAN/tf_WGAN_GP/checkpoints/' + name, force=load_all_ckpt)
        restorer.restore(sess, ckpts[-1])
        size = sample_shape[0] * sample_shape[1]
        #fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/msc_test','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/celeA_List/test_lan_100','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/lan_paper_part','r')
        # path_image = '/home/safe_data_dir_2/guoyu/png_lanzhou_crop/'
        # path_image = "/home/tmp_data_dir/zhaoyu/CelebA/img_align_celeba/"
        #path_image = '/home/xujinchang/share/project/GAN/tf_WGAN_GP/m.0bh0sn/'
        # path_image = "/home/xujinchang/100_128face/"
        # image_list  = []
        # for line in fp.readlines():
        #     image_list.append(line.strip().split(' ')[0])
        # count = 0
        fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/lan_paper_part','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/mean_face/mean_face_msc','r')
        # fp = open('/home/xujinchang/share/project/GAN/tf_WGAN_GP/lan_t1','r')
        # fp = open('/home/tmp_data_dir/malilei/lanzhou_2w_crop/lanzhou_less64_100')
        path_image = '/home/safe_data_dir_2/guoyu/png_lanzhou_crop/'
        # path_image = '/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/'
        image_list  = ['5716.png']
        # for line in fp.readlines():
        #     image_list.append(line.strip().split(' ')[-1])
        # with open('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/display_land3','r')as fp:
        #     imglist = [line.strip().split() for line in fp]
        # image_list = [line[0] for line in imglist]
        # landmarks = [line[1:11] for line in imglist]
        # sum_mean = 0.0
        # sum_var = 0.0
        # factor  = 1
        mean_face = np.zeros((1,64,64,3), dtype=np.float32)
        mean_face2 = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face_msc.jpg',mode='RGB')
        mean_face1 = scipy.misc.imread('/home/xujinchang/share/project/GAN/Face-hallucination-with-tiny-images/mean_face_lans.jpg',mode='RGB')
        for j in range(len(image_list)):
            
            print("count: {}".format(j))
            image = image_list[j]
            print "image: " ,path_image + image_list[j]
            im = scipy.misc.imread(path_image + image_list[j], mode='RGB')
            count = 0
            # mean_value = [np.mean(im), 115.54, 100.54, 120.54]
            # std_value = [np.std(im), 51.29, 60.22]
            # factor1 = [1]   
            factor2 = [0,0.2,0.4,0.5,0.6,0.7,0.8,1.0,2.0]
    
            for i in range(len(factor2)):
                count += 1
                z_, gt, LR, HR = pre_precess_mean(im, mean_face1, factor2[i])
                z_ = z_ / 127.5 - 1.0
                fake_samples = sess.run(model.fake_sample, {model.z: z_})
                fake_samples = (fake_samples + 1.) * 127.5
                merged_samples = utils.merge(fake_samples, size=sample_shape)
                fn = str(count) + '.png'
                scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)
                # count += 1
                # z_, gt, LR, HR = pre_precess_mean(im, mean_face2, factor2[i])
                # z_ = z_ / 127.5 - 1.0
                # fake_samples = sess.run(model.fake_sample, {model.z: z_})
                # fake_samples = (fake_samples + 1.) * 127.5
                # merged_samples = utils.merge(fake_samples, size=sample_shape)
                # fn = str(count) + '.png'
                # scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)

            to_gif(dir_name_sr)
            # z_, gt, LR, HR = pre_precess_lan(im, [128, 128])
            # z_ = z_ / 127.5 - 1.0
            # fake_samples = sess.run(model.fake_sample, {model.z: z_})
            # fake_samples = (fake_samples + 1.) * 127.5
            # merged_samples = utils.merge(fake_samples, size=sample_shape)
            # image_ = image.split('/')[-1]
            # fn = "sr_"+image_
            # scipy.misc.imsave(os.path.join(dir_name_sr, fn), merged_samples)
            # mean_face += HR
            # image = image_list[j]
            # fn_lr = "lr_"+image_
            # fn_hr = "hr_"+image_
            # scipy.misc.imsave(os.path.join(dir_name_lr, fn_lr),LR)
            # scipy.misc.imsave(os.path.join(dir_name_hr, fn_hr),HR)
            
            
            

                
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
        # print "mean: , var: ", sum_mean / len(image_list), sum_var / len(image_list)
        # mean_face = mean_face / len(image_list)
        # mean_face = utils.merge(mean_face, size=sample_shape)
        # mean_face = scipy.misc.imresize(mean_face,[16,16])
        # scipy.misc.imsave(os.path.join(dir_name_lr, 'mean_face'+'.jpg'),mean_face)

'''
You can create a gif movie through imagemagick on the commandline:
$ convert -delay 20 eval/* movie.gif
'''
def to_gif(dir_name='eval'):
    images = []
    im_list = []
    # for path in glob.glob('*.png'):
    #     im_list.append(path) 
    for i in range(9):
        im = scipy.misc.imread(dir_name+'/'+str(i+1)+'.png')
        im = scipy.misc.imresize(im,[256,256])
        images.append(im)

    # make_gif(images, dir_name + '/movie.gif', duration=10, true_image=True)
    imageio.mimsave('movie.gif', images, duration=0.4)

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
