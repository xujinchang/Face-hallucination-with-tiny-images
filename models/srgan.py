# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel
from custom_vgg16 import *
from tensorflow.python.ops import math_ops
from perc_net import *
'''
WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)

+ GP:
Instead of weight clipping, WGAN-GP proposed gradient penalty.
'''
# data_dict = loadWeightsData('/home/xujinchang/share/project/GAN/tf_WGAN_GP/models/vgg16.npy')
# batchsize = 32
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
        # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
    gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32) #AA'
    return gram

# total variation denoising
def total_variation_regularization(x, beta=1):
    assert isinstance(x, tf.Tensor)

    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: tf.nn.conv2d(x, wh, strides=[1, 1, 1, 1], padding='SAME')
    tvw = lambda x: tf.nn.conv2d(x, ww, strides=[1, 1, 1, 1], padding='SAME')
    dh = tvh(x)
    dw = tvw(x)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    return tv


class SRGAN(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[128, 128, 3], z_dim=[32,32,3]):
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.ld = 10. # lambda
        self.n_critic = 1
        self.lambda_f = 1e0
        self.lambda_s = 1e1
        self.lambda_tv = 10e-4
        self.content_loss = 10e-5


        super(SRGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr,
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            # change by xjc z_dim 64 -> [8,8,3]
            X = tf.placeholder(tf.float32, shape=[None] + [128, 128, 3])
            z = tf.placeholder(tf.float32, shape=[None] + [32, 32, 3], name = 'input')
            # import pdb
            # pdb.set_trace()
            global_step = tf.Variable(0, name='global_step', trainable=False)
         
            # vgg_s = custom_Vgg16(X, data_dict=data_dict)

            # feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
            
            # vgg = custom_Vgg16(G, data_dict=data_dict)
            # feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]
            
            # loss_f = tf.zeros(batchsize, tf.float32)
            # for f, f_ in zip(feature, feature_):
            #     loss_f += self.lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

        
            # loss_tv = self.lambda_tv * total_variation_regularization(G)
            # loss = loss_f + loss_tv

            G = self._generator(z)
            D_real_prob, D_real_logits = self._critic(X)
            D_fake_prob, D_fake_logits = self._critic(G, reuse=True)

            G_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_fake_logits), logits=D_fake_logits)
            MSE_loss = tf.losses.mean_squared_error(X, G)

            G_loss = 1e-3*G_loss + MSE_loss

            D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(D_real_logits), logits=D_real_logits)
            D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(D_fake_logits), logits=D_fake_logits)
            D_loss = D_loss_real + D_loss_fake

            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            D_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            with tf.control_dependencies(D_update_ops):
                D_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr, beta1=self.beta1).\
                    minimize(D_loss, var_list=D_vars)
            with tf.control_dependencies(G_update_ops):
                # learning rate 2e-4/1e-3
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1).\
                    minimize(G_loss, var_list=G_vars, global_step=global_step)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar('D_loss', D_loss),
                tf.summary.scalar('D_loss/real', D_loss_real),
                tf.summary.scalar('D_loss/fake', D_loss_fake)
            ])

            # sparse-step summary
           
            tf.summary.histogram('real_probs', D_real_prob)
            tf.summary.histogram('fake_probs', D_fake_prob)

            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('lr_sample', z, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('hr_sample', X, max_outputs=self.FAKE_MAX_OUTPUT)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = D_train_op
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        #return self._good_critic(X, reuse)
        return self._good_critic(X,reuse)
    def _generator(self, z, reuse=False):
        return self._good_generator(z, reuse)
        # return self._good_generator(X,reuse)

    def _leaky_relu(self, x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


    def _good_critic(self, X, reuse=False):
        with tf.variable_scope('critic', reuse=reuse):
            nf = 64
            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], stride=1, padding='SAME', activation_fn=None,  
                weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=None):
                
                net = slim.conv2d(X, nf)
                net = self._leaky_relu(net, 0.2)

                net = slim.conv2d(net, nf, stride=2)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)
# ====================128
                net = slim.conv2d(net, 2*nf)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)
                net = slim.conv2d(net, 2*nf, stride=2)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)
                
# ====================256                                     
                net = slim.conv2d(net, 4*nf)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)
                net = slim.conv2d(net, 4*nf, stride=2)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)

# ====================512  
                net = slim.conv2d(net, 8*nf)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)
                net = slim.conv2d(net, 8*nf, stride=2)
                net = self._leaky_relu(net, 0.2)
                net = slim.batch_norm(net, activation_fn=None, **self.bn_params)
#===================================================================

                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024, activation_fn=None)
                net = self._leaky_relu(net, 0.2)
                net = slim.fully_connected(net, 1, activation_fn=None)

                prob = tf.sigmoid(net)

            return prob, net

                


    def _good_generator(self, X, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
           
            nf = 64
            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], stride=1, padding='SAME', activation_fn=tf.nn.relu,  
                weights_initializer=tf.random_normal_initializer(stddev=0.02), biases_initializer=None):
                
                net = slim.conv2d(X, nf)
                temp = net

                for i in range(16):
                    nn = slim.conv2d(net, nf, activation_fn=None)
                    nn = slim.batch_norm(nn, **self.bn_params)
                    nn = slim.conv2d(nn, nf, activation_fn=None)
                    nn = slim.batch_norm(nn, activation_fn=None, **self.bn_params)
                    nn = net + nn
                    net = nn
                
                net = slim.conv2d(net, nf, activation_fn=None)
                net = slim.batch_norm(net, **self.bn_params)
                net = net + temp
#===================================================================
                
                net = slim.conv2d(net, 4*nf, activation_fn=None)
                bs = tf.shape(net)[0]
                _, h, w, c = net.get_shape().as_list()
                net = tf.transpose(net, (0, 3, 1, 2))
                net = tf.reshape(net, (bs, 2, 2, c // (2 ** 2), h, w))
                net = tf.transpose(net, (0, 3, 4, 1, 5, 2))
                net = tf.reshape(net, (bs, c // (2 ** 2), h * 2, w * 2))
                net = tf.transpose(net, (0, 2, 3, 1))

                net = tf.nn.relu(net)

                net = slim.conv2d(net, 4*nf, activation_fn=None)
                _, h, w, c = net.get_shape().as_list()
                net = tf.transpose(net, (0, 3, 1, 2))
                net = tf.reshape(net, (bs, 2, 2, c // (2 ** 2), h, w))
                net = tf.transpose(net, (0, 3, 4, 1, 5, 2))
                net = tf.reshape(net, (bs, c // (2 ** 2), h * 2, w * 2))
                net = tf.transpose(net, (0, 2, 3, 1))

                net = tf.nn.relu(net)

                net = slim.conv2d(net,3, activation_fn=tf.nn.tanh)    

            return net
