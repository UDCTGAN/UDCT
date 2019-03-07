from __future__ import division, print_function, unicode_literals

import tensorflow as tf

class PatchGAN34:
    """
    This class is creating a PatchGAN discriminator as described by Zhu et al. 2018. 
    -) save()      - Save the current model parameter
    -) create()    - Create the model layers (graph construction)
    -) init()      - Initialize the model (load model if exists)
    -) load()      - Load the parameters from the file
    -) run()       - ToDo write this
    
    Only the following functions should be called from outside:
    -) create()
    -) constructor
    """
    
    def __init__(self,dis_name,noise=0.25):
        """
        Create a PatchGAN model (init). It will check, if a model with such a name has already been saved. If so, the model 
        is being loaded. Otherwise, a new model with this name will be created. It will only be saved, if the save function 
        is being called. The describtion of every parameter is given in the code below.
        
        INPUT: dis_name      - This is the name of the discriminator. It is mainly used to establish the place, where the model 
                               is being saved.
                              
        OUTPUT:              - The model
        """
        self.dis_name         = dis_name
        self.noise            = noise
        
    
    def create(self,X,reuse=True):
        
        # C128
        # To add noise: 
        self.C128_c = tf.layers.conv2d(tf.pad(X+tf.random_normal(tf.shape(X),0.,self.noise),[[0,0],[1,1],[1,1],[0,0]],"Reflect"),
                                  filters=128,
                                  kernel_size=4,
                                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                  strides=(2,2),
                                  padding='valid',
                                  reuse=reuse,
                                  name='dis_'+self.dis_name+'_34_conv_1')
        self.C128_n = tf.contrib.layers.instance_norm(self.C128_c,reuse=reuse,scope='dis_'+self.dis_name+'_34_bnorm_1',trainable=False)
        self.C128   = tf.nn.leaky_relu(self.C128_n)
        
        # C256
        self.C256_c = tf.layers.conv2d(tf.pad(self.C128,[[0,0],[1,1],[1,1],[0,0]],"Reflect"),
                                  filters=256,
                                  kernel_size=4,
                                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                  strides=(2,2),
                                  padding='valid',
                                  reuse=reuse,
                                  name='dis_'+self.dis_name+'_34_conv_2')
        self.C256_n = tf.contrib.layers.instance_norm(self.C256_c,reuse=reuse,scope='dis_'+self.dis_name+'_34_bnorm_2',trainable=False)
        self.C256   = tf.nn.leaky_relu(self.C256_n)
        
        # C512
        self.C512_c = tf.layers.conv2d(tf.pad(self.C256,[[0,0],[1,1],[1,1],[0,0]],"Reflect"),
                                  filters=512,
                                  kernel_size=4,
                                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                  strides=(1,1),
                                  padding='valid',
                                  reuse=reuse,
                                  name='dis_'+self.dis_name+'_34_conv_3')
        self.C512_n = tf.contrib.layers.instance_norm(self.C512_c,reuse=reuse,scope='dis_'+self.dis_name+'_34_bnorm_3',trainable=False)
        self.C512   = tf.nn.leaky_relu(self.C512_n)
        
        # c1
        self.c1_c   = tf.layers.conv2d(tf.pad(self.C512,[[0,0],[1,1],[1,1],[0,0]],"Reflect"),
                                  filters=1,
                                  kernel_size=4,
                                  kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                  strides=(1,1),
                                  padding='valid',
                                  reuse=reuse,
                                  name='dis_'+self.dis_name+'_34_conv_4')
        return self.c1_c