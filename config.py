from models import *


model_zoo = ['DCGAN', 'LSGAN', 'WGAN', 'WGAN-GP', 'EBGAN', 'BEGAN', 'DRAGAN', 'CTGAN', 'AEUGAN', 'PER_GAN']

def get_model(mtype, name, training):
    model = None
    if mtype == 'DCGAN':
        model = dcgan.DCGAN
    elif mtype == 'LSGAN':
        model = lsgan.LSGAN
    elif mtype == 'WGAN':
        model = wgan.WGAN
    elif mtype == 'WGAN-GP':
        model = wgan_gp.WGAN_GP
    elif mtype == 'EBGAN':
        model = ebgan.EBGAN
    elif mtype == 'BEGAN':
        model = began.BEGAN
    elif mtype == 'DRAGAN':
        model = dragan.DRAGAN
    elif mtype == 'CTGAN':
        model = ctgan.CTGAN
    elif mtype == 'AEUGAN':
        model = aeugan.AEUGAN
    elif mtype == 'PER_GAN':
        model = percutal_gan.Per_GAN
    else:
        assert False, mtype + ' is not in the model zoo'

    assert model, mtype + ' is work in progress'

    return model(name=name, training=training)


def get_dataset(dataset_name):
    # celebA_64 = './data/celebA_tfrecords/*.tfrecord'
    # celebA_64 = '/home/xujinchang/GAN_lanc/lan_tfrecords_8_64/*.tfrecord'
    #celebA_64 = '/home/xujinchang/GAN_celeA_LR_HR/celebA_tfrecords_8_64/*.tfrecord'
    # celebA_64 = '/home/xujinchang/GAN_lanc/lanrect_tfrecords_16_64/*.tfrecord'
    # celebA_64 = '/home/tmp_data_dir/xjc/GAN_misc/misc_tfrecords_32_128/*.tfrecord'
    # celebA_64 = '/home/tmp_data_dir/xjc/GAN_misc/mis_bicu_tfrecords_128_128/*.tfrecord'
    # celebA_64 = '/home/xujinchang/GAN_600/600_tfrecords_16_64/*.tfrecord'
    celebA_64 = '/home/xujinchang/GAN_msc/msc_tfrecords_16_64/*.tfrecord'
    # celebA_64 = '/home/tmp_data_dir/xjc/GAN_misc/mis_bicu_tfrecords_128_128/*.tfrecord'
    lsun_bedroom_128 = './data/lsun/bedroom_128_tfrecords/*.tfrecord'

    if dataset_name == 'celeba':
        path = celebA_64
        n_examples = 100000
        # n_examples = 5993250
        #n_examples = 57604
    # if dataset_name == 'celeba':
    #     path = celebA_64
    #     n_examples = 202599
    # if dataset_name == 'celeba':
    #    path = celebA_64
    #    n_examples = 40000
    elif dataset_name == 'lsun':
        path = lsun_bedroom_128
        n_examples = 3033042
    else:
        raise ValueError('{} is does not supported. dataset must be celeba or lsun.'.format(dataset_name))

    return path, n_examples


def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")

