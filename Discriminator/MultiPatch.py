from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import PatchGAN34
import PatchGAN70
import PatchGAN142

class MultiPatch:
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
        
        self.Patch34          = PatchGAN34.PatchGAN34(self.dis_name,noise=self.noise)
        self.Patch70          = PatchGAN70.PatchGAN70(self.dis_name,noise=self.noise)
        self.Patch142         = PatchGAN142.PatchGAN142(self.dis_name,noise=self.noise)
    
    def create(self,X,reuse=True):
        
        self.out34            = self.Patch34.create(X,reuse)
        self.out70            = self.Patch70.create(X,reuse)
        self.out142           = self.Patch142.create(X,reuse)
        
        reshaped34            = tf.reshape(self.out34,[-1,tf.shape(self.out34)[1]*tf.shape(self.out34)[2],1])
        reshaped70            = tf.reshape(self.out70,[-1,tf.shape(self.out70)[1]*tf.shape(self.out70)[2],1])
        reshaped142           = tf.reshape(self.out142,[-1,tf.shape(self.out142)[1]*tf.shape(self.out142)[2],1])
        
        self.prediction       = tf.concat([reshaped34,reshaped70,reshaped142],axis=1)
        return self.prediction