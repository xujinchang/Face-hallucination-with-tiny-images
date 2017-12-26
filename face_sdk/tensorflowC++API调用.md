## 第一步  保存图和模型
```
    在训练模型的时候，除了保存模型checkpoint 之外，还要保存graph
    Tensorflow 在前端 Python 中构建图，并且通过将该图序列化到 ProtoBuf GraphDef，以方便在后端运行        
            
    def train(model, input_op, num_epochs, batch_size, n_examples, renew=False):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer()) # for epochs
            
            saver = tf.train.Saver(max_to_keep=10)
            batch_X, batch_z = sess.run(input_op)
            
            _ = sess.run([model.G_train_op], feed_dict={model.X: batch_X, model.z: batch_X_z})
            
            
            # 存图
            tf.train.write_graph(sess.graph_def, ckpt_path, "nn_model.pbtxt", as_text=True)
            # 存模型文件
            saver.save(sess, ckpt_path+'/'+model.name, global_step=global_step)
            #checkpoint 模型的轮数 
            #.data:存放的是权重参数 
            #全面保存了训练某时间截面的信息，包括参数，超参数，梯度等等
            #ckpt文件保存了Variable的二进制信息，index文件用于保存 ckpt  
            #.index 文件中对应 Variable 的偏移量信息
            #.meta:存放的是图和metadata,metadata是其他配置的数据
            #如果想将我们的模型固化，让别人能够使用，我们仅仅需要的是图和
            #参数，metadata是不需要的
   ```         
## 第二步 将该模型的图结构和该模型的权重固化到一起了

### 2.1 freeze
```
    利用tensorflow自带的freeze_graph.py小工具把.ckpt文件中的参数固定在graph内，输出nn_model_frozen.pb
    
    python freeze_graph.py 
    --input_graph=../model/nn_model.pbtxt  模型的图的定义文件
    --input_checkpoint=../ckpt/nn_model.ckpt .data 之前的模型的参数文件
    --output_graph=../model/nn_model_frozen.pb 绑定后包含参数的图模型文件
    --output_node_names=output_node  输出待计算的tensor名字
    
    python freeze.py --input_graph=./nn_model.pbtxt 
    --input_checkpoint=./WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc-620000
    --output_graph=./nn_model_frozen.pb 
    --output_node_names=WGAN-GP_model_16_64_ps_noise_near_conv_l2_msc/generator/Conv/Tanh
  
    对于freeze操作,我们需要定义输出结点的名字.因为网络其实是比较复杂的,
    定义了输出结点的名字,那么freeze的时候就只把输出该结点所需要的子图都固
    化下来,其他无关的就舍弃掉.因为我们freeze模型的目的是接下来做预测.所以,
    一般情况下,output_node_names就是我们预测的目标        
``` 
### 2.2 pb
```
def load_graph(frozen_graph_filename):  
    # We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
  
    # We load the graph_def in the default graph  
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
```
## 第三步 C++API调用模型和编译

### 3.1 bazel 安装和配置
```
    首先安装bazel，https://bazel.build/versions/master/docs/install-compile-source.html，
    在编译Bazel 时候必须安装了jdk, 我这里安装的是jdk-8.之后，java 的版本必须高于7 
    下载 bazel-<VERSION>-dist.zip unzip  之后, 执行 bash ./compile.sh 会自动编译,
    编译结束后, 将生成的output/bazel 的添加到你的PATH 环境变量里面去
    
    坑：服务器的base_source 里面的local/bin 里面也有一个bazel， 所以我们要将新遍的bazel的文件路径加到PATH的最前面，于是我将base_source 里面的给拿到.zshrc 中，这样新编译的bazel才能正常使用
    
    export PATH=$PATH:/home/zhaoyu/share/software/DrMemory-Linux-1.10.1-3/bin64 
    export JAVA_PATH=/home/software/java/jdk1.8.0_101
    export JRE_HOME=$JAVA_PATH/jre
    export PATH=$JAVA_PATH/bin:$PATH
    export CLASSPATH=.:$JAVA_PATH/lib
    export CLASSPATH=$CLASSPATH:/home/xujinchang/share/project/algrithom/algs4/algs4.jar
    export PATH=$PATH:/home/zhaoyu/share/software/imgcat
    export PATH=/home/xujinchang/share/download/baze/output/bazel:/home/software/java/jdk1.8.0_101/bin:/home/xujinchang/share/geeqie:$PATH
    export JAVA_HOME=/home/software/java/jdk1.8.0_101
    
```
### 3.2 bazel tutorial

```
    需要build， build 里面写的是要编译的源文件，头文件，依赖等
    bazel build //path/to/target:target-name
    指定的name cc_library 编库，里面的参数如下
    cc_library(
      name = "faceall",
      srcs = ["faceall.cc","imageio.cc"],
      hdrs= ["faceall.h","imageio.h"],
      deps = [
                  "//tensorflow/cc:cc_ops",
                  "//tensorflow/cc:client_session",
                  "//tensorflow/core:tensorflow",
                  "//tensorflow/cc:scope",
              ],
          alwayslink = 1,
    )
    
    编二进制文件
    cc_binary(
        name = "test_faceall",
        srcs = ["test_faceall.cc"],
        deps = [
                    "//tensorflow/cc/faceall:faceall",
                ],
    )
    
    bazel build -c dbg --copt="-fPIC" //tensorflow/cc/faceall:faceall

    bazel build -c dbg --copt="-fPIC" //tensorflow/cc/faceall:test_faceall
    
    在 bazel/bin目录下tensorflow/cc/faceall目录下即可生成动态库和测试的test_faceall文件
    
```
## 第四步 测试
```
    进入到bazel-bin/tensorflow/cc/faceall 中
    将.pb ori.jpg 拷过来
    执行./test_faceall 就可以得到output.jpg
```

# 相关链接
- https://zhuanlan.zhihu.com/p/31308381
- https://www.jianshu.com/p/b6f9451716ed?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation 读取
- http://blog.csdn.net/c2a2o2/article/details/72778628
- https://github.com/smuelpeng/tensorflow_cpp_sdk_tutorial

            
            
            
            
            
            
            

            
