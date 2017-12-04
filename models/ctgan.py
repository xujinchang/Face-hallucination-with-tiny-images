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

class CTGAN(BaseModel):
    def __init__(self, name, training, D_lr=1e-4, G_lr=1e-4, image_shape=[64, 64, 3], z_dim=[16,16,3]):
        self.beta1 = 0.0
        self.beta2 = 0.9
        self.ld = 10. # lambda
        self.n_critic = 5
        super(CTGAN, self).__init__(name=name, training=training, D_lr=D_lr, G_lr=G_lr, 
            image_shape=image_shape, z_dim=z_dim)

    def _build_train_graph(self):
        with tf.variable_scope(self.name):
            # change by xjc z_dim 64 -> [8,8,3]
            X = tf.placeholder(tf.float32, [None] + self.shape)
            tm = tf.placeholder(tf.float32, [None] + self.shape)
            z = tf.placeholder(tf.float32, [None] + self.z_dim)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # `critic` named from wgan (wgan-gp use the term `discriminator` rather than `critic`)

            G0, G1, G2 = self._generator(z, tm)
            C_real = self._critic(X)
            C_fake0 = self._critic(G0, reuse=True)
            C_fake1 = self._critic(G1, reuse=True)
            C_fake2 = self._critic(G2, reuse=True)

            W_dist0 = 1*tf.reduce_mean(C_real - C_fake0)
            W_dist1 = 1*tf.reduce_mean(C_real - C_fake1)
            W_dist2 = 3*tf.reduce_mean(C_real - C_fake2)

            W_dist = (W_dist0 + W_dist1 + W_dist2) / (1+1+3)

            # fea_X_norm = tf.divide(fea_X, tf.norm(fea_X, ord = 'euclidean'))
            # fea_G_norm = tf.divide(fea_G, tf.norm(fea_G, ord = 'euclidean'))
            # L2_dist = tf.sqrt(tf.reduce_sum(tf.square(fea_X - fea_G)))   #(batch,1)
            
            # tf.norm(slim.flatten(C_xhat_grad), axis=1)p
            
            # gen_l1_cost = tf.reduce_mean(tf.abs(G - X))
            gen_l2_cost = tf.sqrt(tf.reduce_sum(tf.square(G0 - X))) + tf.sqrt(tf.reduce_sum(tf.square(G1 - X))) + 3*tf.sqrt(tf.reduce_sum(tf.square(G2 - X)))
            C_loss = -W_dist
            #- L2_dist
            
            G_loss = 0.1 * (tf.reduce_mean(-C_fake0) + tf.reduce_mean(-C_fake1) + 3*tf.reduce_mean(-C_fake2)) / 5 + 0.9 * gen_l2_cost / 5
            # add by xjc MSE_loss
            # MSE_loss = tf.reduce_mean(slim.losses.mean_squared_error(predictions=G, labels=X, weights=1.0)) 
            # G_loss += MSE_loss
            # Gradient Penalty (GP)
            eps = tf.random_uniform(shape=[tf.shape(X)[0], 1, 1, 1], minval=0., maxval=1.)
            x_hat = eps*X + (1.-eps)*(G0 + G1 + 3*G2) / 5 
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
                tf.summary.scalar(' L2_dist', gen_l2_cost),
                # tf.summary.scalar('MSE_loss', MSE_loss),
                tf.summary.scalar('C_loss', C_loss),
                tf.summary.scalar('W_dist', W_dist),
                tf.summary.scalar('GP', GP)
            ])

            # sparse-step summary
            tf.summary.image('fake_sample', G2, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('lr_sample', z, max_outputs=self.FAKE_MAX_OUTPUT)
            tf.summary.image('hr_sample', X, max_outputs=self.FAKE_MAX_OUTPUT)
            # tf.summary.histogram('real_probs', D_real_prob)
            # tf.summary.histogram('fake_probs', D_fake_prob)
            self.all_summary_op = tf.summary.merge_all()

            # accesible points
            self.X = X
            self.tm = tm
            self.z = z
            self.D_train_op = C_train_op # train.py 와의 accesibility 를 위해... 흠... 구린데...
            self.G_train_op = G_train_op
            self.fake_sample = G2
            self.global_step = global_step

    def _critic(self, X, reuse=False):
        #return self._good_critic(X, reuse)
        return self._good_critic(X,reuse)
    def _generator(self, z, net_tm0, reuse=False):
        return self._good_generator(z, net_tm0, reuse)
        # return self._good_generator(X,reuse)
  
    def _residual_block(self, X, nf_output, resample, kernel_size=[3,3], name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
            nf_input = input_shape[-1]

            if resample == 'down': # Downsample
                shortcut = slim.avg_pool2d(X, [2,2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) # init xavier

                net = slim.layer_norm(X, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None) # skip bias
                net = slim.layer_norm(net, activation_fn=tf.nn.relu)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
                net = slim.avg_pool2d(net, [2,2])

                return net + shortcut
            elif resample == 'up': # Upsample
                # _x = slim.conv2d(X, nf_output*4, kernel_size=[1,1], activation_fn=None)
                input_shape = X.shape
                nf_input = input_shape[-1]
                _x = X
                bs = tf.shape(_x)[0]
                r = 2
                upsample_shape = map(lambda x: int(x)*2, input_shape[1:3])
                shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape) #dui x chaongfu 
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) #ke xuan xiang
                

                _net = slim.batch_norm(X, activation_fn=tf.nn.relu, **self.bn_params)
                _net = slim.conv2d(_net, nf_output*4, kernel_size=[3,3], activation_fn=None)
                _, h, w, c = _net.get_shape().as_list()
                _net = tf.transpose(_net, (0, 3, 1, 2))
                _net = tf.reshape(_net, (bs, r, r, c // (r ** 2), h, w))
                _net = tf.transpose(_net, (0, 3, 4, 1, 5, 2))
                _net = tf.reshape(_net, (bs, c // (r ** 2), h * 2, w * 2))
                _net = tf.transpose(_net, (0, 2, 3, 1))
                # _net = slim.batch_norm(_net, activation_fn=tf.nn.relu, **self.bn_params)
                # _net = slim.conv2d(_net, nf_output, kernel_size=[5,5], biases_initializer=None) # skip bias
                _net = slim.batch_norm(_net, activation_fn=tf.nn.relu, **self.bn_params)
                _net = slim.conv2d(_net, nf_output, kernel_size=[5,5])

                return _net + shortcut
            elif resample == "same":
                _net = slim.batch_norm(X, activation_fn=tf.nn.relu, **self.bn_params)
                _net = slim.conv2d(_net, nf_output, kernel_size=kernel_size, biases_initializer=None) # skip bias
                _net = slim.batch_norm(_net, activation_fn=tf.nn.relu, **self.bn_params)
                _net = slim.conv2d(_net, nf_output, kernel_size=kernel_size)
                return X + _net
            else:
                raise Exception('invalid resample value')
    

    def _good_critic(self, X, reuse=False):
        with tf.variable_scope('critic', reuse=reuse):
            nf = 64
            net = slim.conv2d(X, nf, [3,3], activation_fn=None) # 64x64x64
            net = self._residual_block(net, 2*nf, resample='down', name='res_block1') # 32x32x128
            net = self._residual_block(net, 4*nf, resample='down', name='res_block2') # 16x16x256
            net = self._residual_block(net, 8*nf, resample='down', name='res_block3') # 8x8x512
            net = self._residual_block(net, 8*nf, resample='down', name='res_block4') # 4x4x512
            # expected_shape(net, [4, 4, 512])
            net = slim.conv2d(net, 1, [1,1], activation_fn=None)
            net = slim.avg_pool2d(net, [4,4])
            # add by xjc  add a bottleneck layer
            net = slim.flatten(net)
        
            return net

    def _good_generator(self, lr, net_tm0, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            nf = 64

            h_lr = self._encoder_lr(lr, reuse=reuse) # extract lr feature
            import pdb
            pdb.set_trace()
            h_tm0 = self._encoder_tm(net_tm0, reuse=reuse) # extract template feature

            h_combine0 = tf.concat([h_lr, h_tm0], 1) # 128
            
            net_tm1 = self._decoder(h_combine0, reuse=reuse) # 64*64*3
            h_tm1 = self._encoder_tm(net_tm1, reuse=True) # template 2
           
            h_combine1 = tf.concat([h_lr, h_tm1], 1) # 128
            
            net_tm2 = self._decoder(h_combine1, reuse=True)
            h_tm2 = self._encoder_tm(net_tm2, reuse=True)
            
            h_combine2 = tf.concat([h_lr, h_tm2], 1)

            net_tm3 = self._decoder(h_combine2, reuse=True)

            return net_tm1, net_tm2, net_tm3
        
            # # expected_shape(net, [64, 64, 64])
            # net = slim.batch_norm(net, activation_fn=tf.nn.relu, **self.bn_params)
            # net = slim.conv2d(net, 3, kernel_size=[3,3], activation_fn=tf.nn.tanh)
            # # expected_shape(net, [64, 64, 3])

            # return net
    # for 16*16 LR
    def _encoder_lr(self, X, reuse=False):
        with tf.variable_scope('encoder_lr', reuse=reuse):
            nf = 64
            nh = 64

            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(X, nf)

                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*4, stride=2) # 8x8

                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4, stride=2) # 4*4

            net = slim.flatten(net)
            h = slim.fully_connected(net, nh, activation_fn=None) # 64
            return h
    
    # for 64*64 examplr
    def _encoder_tm(self, X, reuse=False):
        with tf.variable_scope('encoder_tm', reuse=reuse):
            nf = 64
            nh = 64
            import pdb
            pdb.set_trace()
            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
                net = slim.conv2d(X, nf)
                
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf*2, stride=2) # 32x32

                net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*2)
                net = slim.conv2d(net, nf*3, stride=2) # 16x16

                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*3)
                net = slim.conv2d(net, nf*4, stride=2) # 8x8

                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4)
                net = slim.conv2d(net, nf*4, stride=2) # 4*4

            net = slim.flatten(net)
            h = slim.fully_connected(net, nh, activation_fn=None) # 64
            return h

    # generator
    def _decoder(self, h, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            nf = 64
            # nh = self.z_dim

            h0 = slim.fully_connected(h, 4*4*4*nf, activation_fn=None) # h0
            net = tf.reshape(h0, [-1, 4, 4, 4*nf])
            with slim.arg_scope([slim.conv2d], kernel_size=[3,3], padding='SAME', activation_fn=tf.nn.elu):
               
                net = slim.conv2d(net, 4*nf)
                net = slim.conv2d(net, 4*nf)
                net = tf.image.resize_nearest_neighbor(net, [8, 8]) # upsampling

                net = slim.conv2d(net, 4*nf)
                net = slim.conv2d(net, 4*nf)
                net = tf.image.resize_nearest_neighbor(net, [16, 16]) # upsampling

                net = slim.conv2d(net, 3*nf)
                net = slim.conv2d(net, 3*nf)
                net = tf.image.resize_nearest_neighbor(net, [32, 32])

                net = slim.conv2d(net, 2*nf)
                net = slim.conv2d(net, 2*nf)
                net = tf.image.resize_nearest_neighbor(net, [64, 64])

                net = slim.conv2d(net, nf)
                net = slim.conv2d(net, nf)

                net = slim.conv2d(net, 3, activation_fn=tf.nn.tanh)

            return net
