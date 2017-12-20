# coding: utf-8
import tensorflow as tf
slim = tf.contrib.slim
from utils import expected_shape
import ops
from basemodel import BaseModel

'''
WGAN:
WD = max_f [ Ex[f(x)] - Ez[f(g(z))] ] where f has K-Lipschitz constraint
J = min WD (G_loss)

+ GP:
Instead of weight clipping, WGAN-GP proposed gradient penalty.
'''

class AEUGAN(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[128, 128, 3], z_dim=[128,128,3]):
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.ld = 10. # lambda
        self.n_critic = 1
        super(WGAN_GP, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr,
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            # change by xjc z_dim 64 -> [8,8,3]
            X = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None] + self.z_dim)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # `critic` named from wgan (wgan-gp use the term `discriminator` rather than `critic`)

            G = self._generator(z)
            C_real = self._critic(X)
            C_fake = self._critic(G, reuse=True)

            W_dist = tf.reduce_mean(C_real - C_fake)



            gen_l1_cost = tf.reduce_mean(tf.abs(G - X))
            gen_l2_cost = tf.sqrt(tf.reduce_sum(tf.square(G - X)))
            C_loss = -W_dist
            #- L2_dist

            G_loss = 0.1 * tf.reduce_mean(-C_fake) + 0.9 * gen_l2_cost
            
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = eps*X + (1.-eps)*G
            C_xhat = self._critic(x_hat, reuse=True)
            C_xhat_grad = tf.gradients(C_xhat, x_hat)[0] # gradient of D(x_hat)
            C_xhat_grad_norm = tf.norm(slim.flatten(C_xhat_grad), axis=1)  # l2 norm
            GP = self.ld * tf.reduce_mean(tf.square(C_xhat_grad_norm - 1.))
            C_loss += GP

            C_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/critic/')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/generator/')

            C_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/critic/')
            G_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name+'/generator/')

            n_critic = 5
            lr = 1e-4
            with tf.control_dependencies(C_update_ops):
                C_train_op = tf.train.AdamOptimizer(learning_rate=self.D_lr*n_critic, beta1=self.beta1, beta2=self.beta2).\
                    minimize(C_loss, var_list=C_vars, global_step=global_step)
            with tf.control_dependencies(G_update_ops):
                G_train_op = tf.train.AdamOptimizer(learning_rate=self.G_lr, beta1=self.beta1, beta2=self.beta2).\
                    minimize(G_loss, var_list=G_vars)

            # summaries
            # per-step summary
            self.summary_op = tf.summary.merge([
                tf.summary.scalar('G_loss', G_loss),
                tf.summary.scalar(' L1_dist', gen_l1_cost),
                tf.summary.scalar(' L2_dist', gen_l2_cost),
                # tf.summary.scalar('MSE_loss', MSE_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('W_dist', W_dist),
                tf.summary.scalar('GP', GP)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('lr_sample', z, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('hr_sample', X, max_outputs=self.FAKE_MAX_OUTPUT)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.z = z
            self.D_train_op = C_train_op # train.py 와의 accesibility 를 위해... 흠... 구린데...
            self.G_train_op = G_train_op
            self.fake_sample = G
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        #return self._good_critic(X, reuse)
        return self._good_critic(X,reuse)
    def _generator(self, z, reuse=False):
        return self._good_generator(z, reuse)
        # return self._good_generator(X,reuse)



#AutoDecoder with Unet and WGAN-GP
    def _good_generator(self, X, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            nf = 64
            # nh = self.z_dim
            # import pdb
            # pdb.set_trace()
            with slim.arg_scope([slim.conv2d], kernel_size=[4,4], stride = 2, padding='SAME', activation_fn=tf.nn.relu):
                
                dw_h_convs0 = slim.conv2d(X, nf) #64x64x64
                dw_h_convs1 = slim.conv2d(dw_h_convs0, nf*2) # 32x32x128
                dw_h_convs2 = slim.conv2d(dw_h_convs1, nf*4) # 16x16x256
                dw_h_convs3 = slim.conv2d(dw_h_convs2, nf*8) # 8x8x512
                dw_h_convs4 = slim.conv2d(dw_h_convs3, nf*8) # 4x4x512
                dw_h_convs5 = slim.conv2d(dw_h_convs4, nf*8) # 2x2x512
                
            net = slim.flatten(dw_h_convs5)
            h = slim.fully_connected(net, 2*nf, activation_fn=None)
            h0 = slim.fully_connected(h, 2*2*8*nf, activation_fn=None) # h0
            net = tf.reshape(h0, [-1, 2, 2, 8*nf])
            
            with slim.arg_scope([slim.conv2d_transpose], kernel_size=[4,4], stride = 2, padding='SAME', activation_fn=tf.nn.relu):
                up_h_convs0 = slim.conv2d_transpose(net, nf*8)
                up_h_convs0 = tf.concat([dw_h_convs4, up_h_convs0], 3) # 4x4x1024

                up_h_convs1 = slim.conv2d_transpose(up_h_convs0, nf*8)
                up_h_convs1 = tf.concat([dw_h_convs3, up_h_convs1], 3) # 8x8x1024

                up_h_convs2 = slim.conv2d_transpose(up_h_convs1, nf*4)
                up_h_convs2 = tf.concat([dw_h_convs2, up_h_convs2], 3) # 16x16x512

                up_h_convs3 = slim.conv2d_transpose(up_h_convs2, nf*2)
                up_h_convs3 = tf.concat([dw_h_convs1, up_h_convs3], 3) # 32x32x256
                
                up_h_convs4 = slim.conv2d_transpose(up_h_convs3, nf)
                up_h_convs4 = tf.concat([dw_h_convs0, up_h_convs4], 3) # 64x64x128
                

                net = slim.conv2d_transpose(up_h_convs4, 3, activation_fn=tf.nn.tanh)

            # net = slim.flatten(net)
            # h = slim.fully_connected(net, nh, activation_fn=None)
            #add a 1*1 conv
           

            return net

    def _good_critic(self, X, reuse=False):
        with tf.variable_scope('critic', reuse=reuse):
            nf = 64
           
            with slim.arg_scope([slim.conv2d], kernel_size=[4,4], stride = 2, padding='SAME', activation_fn=tf.nn.relu):
                
                # import pdb
                # pdb.set_trace()
                dw_h_convs0 = slim.conv2d(X, nf) #64x64x64
                dw_h_convs1 = slim.conv2d(dw_h_convs0, nf*2) # 32x32x128
                dw_h_convs2 = slim.conv2d(dw_h_convs1, nf*4) # 16x16x256
                dw_h_convs3 = slim.conv2d(dw_h_convs2, nf*8) # 8x8x512
                dw_h_convs4 = slim.conv2d(dw_h_convs3, nf*8) # 4x4x512

                net = slim.flatten(dw_h_convs4)
                net = slim.fully_connected(net, 1, activation_fn=None)

            return net


   