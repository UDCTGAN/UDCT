from __future__ import division, print_function, unicode_literals

import tensorflow as tf

class HisDis:
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
    
    def __init__(self,dis_name,noise=0.1,keep_prob=0.5):
        """
        Create a histogramm discriminator
        
        INPUT: dis_name      - This is the name of the discriminator. It is mainly used to establish the place, where the model 
                               is being saved.
                              
        OUTPUT:              - The model
        """
        self.dis_name         = dis_name
        self.noise            = noise
        self.keep_prob        = keep_prob
        
    
    def create(self,X,reuse=True):
        """
        Create a histogramm discriminator
        
        INPUT: X             - [None,256*n_chan]
                              
        OUTPUT:              - The HisDis prediction
        """
        self.hidden_1  = tf.layers.dense(tf.nn.dropout(X-0.5+tf.random_normal(tf.shape(X),0.,self.noise),keep_prob=self.keep_prob),
                                         64,
                                         reuse=reuse,
                                         name='dis_'+self.dis_name+'_hidden_1',
                                         activation=tf.nn.tanh)
        
        self.hidden_2  = tf.layers.dense(tf.nn.dropout(self.hidden_1,keep_prob=self.keep_prob),
                                         64,
                                         reuse=reuse,
                                         name='dis_'+self.dis_name+'_hidden_2',
                                         activation=tf.nn.tanh)
        
        self.out       = tf.layers.dense(tf.nn.dropout(self.hidden_2,keep_prob=self.keep_prob),
                                         1,
                                         reuse=reuse,
                                         name='dis_'+self.dis_name+'_h_out',
                                         activation=None) + 0.5
        
        return self.out
    
