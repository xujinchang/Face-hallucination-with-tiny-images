# Face-hallucination-with-tiny-images
using gan to super-resolve tiny images with an upscaling factor of 8

<img src='display/1.jpg' >
<img src='display/2.jpg' >


# Denoising Face Images

<img src='display/noise1.png'>
<img src='display/noise2.png'>


## Usage

 cited in [xujinchang/tf.gans-comparison](https://github.com/xujinchang/tf.gans-comparison)
 
 Download CelebA dataset:
 
 ```
 $ python download.py celeba
 ```
 
 Convert images to tfrecords format. Options for converting are hard-coded, so ensure to modify it before run `convert.py`.
 
 ```
 $ python convert.py
 ```
 
 Train. If you want to change the settings of each model, you must also modify code directly.
 
 ```
 $ python train.py --help
 usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                 [--num_threads NUM_THREADS] --model MODEL [--name NAME]
                 [--renew]
 
 optional arguments:
   -h, --help            show this help message and exit
   --num_epochs NUM_EPOCHS
                         default: 20
   --batch_size BATCH_SIZE
                         default: 128 
   --num_threads NUM_THREADS
                         # of data read threads (default: 4)
   --model MODEL         DCGAN / LSGAN / WGAN / WGAN-GP / EBGAN / BEGAN /
                         DRAGAN
   --name NAME           default: name=model
   --renew               train model from scratch - clean saved checkpoints and 
                         summaries
 ```
 
 Monitor through TensorBoard:
 
 ```
 $ tensorboard --logdir=summary/name
```
 
 Evaluate (generate fake samples):
 
 ```
 $ python eval.py --help
 usage: eval.py [-h] --model MODEL [--name NAME]
 
 optional arguments:
   -h, --help     show this help message and exit
   --model MODEL  DCGAN / LSGAN / WGAN / WGAN-GP / EBGAN / BEGAN / DRAGAN
   --name NAME    default: name=model
 ```
 
 
 ### Requirements
 
 - python 2.7
 - tensorflow 1.2
 - tqdm
 - (optional) pynvml - for automatic gpu selection
 







 

