from __future__ import division, print_function, unicode_literals


import tensorflow as tf

import h5py
import numpy as np

import os

import sys
sys.path.append('./Discriminator')
sys.path.append('./Generator')
sys.path.append('./Utilities/')
import Res_Gen
import PatchGAN34
import PatchGAN70
import PatchGAN142
import MultiPatch
import HisDis
import Utilities
import cv2
class Model:
    """
    ToDo
    -) save()      - Save the current model parameter
    -) create()    - Create the model layers
    -) init()      - Initialize the model (load model if exists)
    -) load()      - Load the parameters from the file
    -) ToDo
    
    Only the following functions should be called from outside:
    -) ToDo
    -) constructor
    """
    
    def __init__(self,
                 mod_name,
                 data_file,
                 buffer_size=32,
                 architecture='Res6',
                 lambda_h=10.,\
                 lambda_c=10.,\
                 dis_noise=0.25,\
                 deconv='transpose',\
                 patchgan='Patch70',\
                 verbose=False,\
                 gen_only=False):
        """
        Create a Model (init). It will check, if a model with such a name has already been saved. If so, the model is being 
        loaded. Otherwise, a new model with this name will be created. It will only be saved, if the save function is being 
        called. The describtion of every parameter is given in the code below.
        
        INPUT: mod_name      - This is the name of the model. It is mainly used to establish the place, where the model is being 
                               saved.
               data_file     - hdf5 file that contains the dataset
               imsize        - The dimension of the input images
                              
        OUTPUT:             - The model
        """
        
        self.mod_name            = mod_name                               # Model name (see above)
        
        self.data_file           = data_file                              # hdf5 data file
        
        f = h5py.File(self.data_file,"r")
        self.a_chan              = int(np.array(f['A/num_channel']))      # Number channels in A
        self.b_chan              = int(np.array(f['B/num_channel']))      # Number channels in B
        self.imsize              = int(np.shape(f['A/data'][0,:,0,0])[0]) # Image size (squared)
        self.a_size              = int(np.array(f['A/num_samples']))      # Number of samples in A
        self.b_size              = int(np.array(f['B/num_samples']))      # Number of samples in B
        f.close()
                
        # Reset all current saved tf stuff
        tf.reset_default_graph()
        
        self.architecture        = architecture
        self.lambda_h            = lambda_h
        self.lambda_c            = lambda_c
        self.dis_noise_0         = dis_noise                              # ATTENTION: Name change from dis_noise to dis_noise_0
        self.deconv              = deconv
        self.patchgan            = patchgan
        self.verbose             = verbose
        self.gen_only            = gen_only  # If true, only the generator are used (and loaded)
        
        # Create the model that is built out of two discriminators and a generator
        self.create()
        
        # Image buffer
        self.buffer_size         = buffer_size
        self.temp_b_s            = 0.
        self.buffer_real_a       = np.zeros([self.buffer_size,self.imsize,self.imsize,self.a_chan])
        self.buffer_real_b       = np.zeros([self.buffer_size,self.imsize,self.imsize,self.b_chan])
        self.buffer_fake_a       = np.zeros([self.buffer_size,self.imsize,self.imsize,self.a_chan])
        self.buffer_fake_b       = np.zeros([self.buffer_size,self.imsize,self.imsize,self.b_chan])
        
        # Create the model saver
        with self.graph.as_default():
            if not self.gen_only:
                self.saver    = tf.train.Saver()
            else:
                self.saver    = tf.train.Saver(var_list=self.list_gen)
    
    def create(self):
        """
        Create the model. ToDo
        """
        # Create a graph and add all layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define variable learning rate and dis_noise
            self.relative_lr    = tf.placeholder_with_default([1.],[1],name="relative_lr")
            self.relative_lr    = self.relative_lr[0]
            
            self.rel_dis_noise  = tf.placeholder_with_default([1.],[1],name="rel_dis_noise")
            self.rel_dis_noise  = self.rel_dis_noise[0]
            self.dis_noise      = self.rel_dis_noise * self.dis_noise_0
            
            
            # Create the generator and discriminator
            if self.architecture == 'Res6':
                                gen_dim =    [64,   128,256,   256,256,256,256,256,256,   128,64     ]
                                kernel_size =[7,    3,3,       3,3,3,3,3,3,               3,3,      7]
            elif self.architecture == 'Res9':
                                gen_dim=    [64,   128,256,   256,256,256,256,256,256,256,256,256,   128,64    ]
                                kernel_size=[7,    3,3,       3,3,3,3,3,3,3,3,3,                     3,3,     7]
            else:
                                print('Unknown generator architecture')
                                return None
            
            self.genA           = Res_Gen.ResGen('BtoA',self.a_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            self.genB           = Res_Gen.ResGen('AtoB',self.b_chan,gen_dim=gen_dim,kernel_size=kernel_size,deconv=self.deconv,verbose=self.verbose)
            
            if self.patchgan == 'Patch34':
                self.disA       = PatchGAN34.PatchGAN34('A',noise=self.dis_noise)
                self.disB       = PatchGAN34.PatchGAN34('B',noise=self.dis_noise)
            elif self.patchgan == 'Patch70':
                self.disA       = PatchGAN70.PatchGAN70('A',noise=self.dis_noise)
                self.disB       = PatchGAN70.PatchGAN70('B',noise=self.dis_noise)
            elif self.patchgan == 'Patch142':
                self.disA       = PatchGAN142.PatchGAN142('A',noise=self.dis_noise)
                self.disB       = PatchGAN142.PatchGAN142('B',noise=self.dis_noise)
            elif self.patchgan == 'MultiPatch':
                self.disA       = MultiPatch.MultiPatch('A',noise=self.dis_noise)
                self.disB       = MultiPatch.MultiPatch('B',noise=self.dis_noise)
            else:
                print('Unknown Patch discriminator type')
                return None
            
            self.disA_His   = HisDis.HisDis('A',noise=self.dis_noise,keep_prob=1.)
            self.disB_His   = HisDis.HisDis('B',noise=self.dis_noise,keep_prob=1.)
        
            # Create a placeholder for the input data
            self.A           = tf.placeholder(tf.float32,[None, None, None, self.a_chan],name="a")
            self.B           = tf.placeholder(tf.float32,[None, None, None, self.b_chan],name="b")
            
            if self.verbose:
                print('Size A: ' +str(self.a_chan)) # Often 1 --> Real
                print('Size B: ' +str(self.b_chan)) # Often 3 --> Syn
            
            # Create cycleGAN                
            
            self.fake_A      = self.genA.create(self.B,False)
            self.fake_B      = self.genB.create(self.A,False)
            
            
            
            # Define the histogram loss
            t_A             = tf.transpose(tf.reshape(self.A,[-1, self.a_chan]),[1,0])
            t_B             = tf.transpose(tf.reshape(self.B,[-1, self.b_chan]),[1,0])
            t_fake_A        = tf.transpose(tf.reshape(self.fake_A,[-1, self.a_chan]),[1,0])
            t_fake_B        = tf.transpose(tf.reshape(self.fake_B,[-1, self.b_chan]),[1,0])

            self.s_A,_      = tf.nn.top_k(t_A,tf.shape(t_A)[1])
            self.s_B,_      = tf.nn.top_k(t_B,tf.shape(t_B)[1])
            self.s_fake_A,_ = tf.nn.top_k(t_fake_A,tf.shape(t_fake_A)[1])
            self.s_fake_B,_ = tf.nn.top_k(t_fake_B,tf.shape(t_fake_B)[1])
            
            self.m_A        = tf.reshape(tf.reduce_mean(tf.reshape(self.s_A,[self.a_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_B        = tf.reshape(tf.reduce_mean(tf.reshape(self.s_B,[self.b_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_fake_A   = tf.reshape(tf.reduce_mean(tf.reshape(self.s_fake_A,[self.a_chan, self.imsize, -1]),axis=2),[1, -1])
            self.m_fake_B   = tf.reshape(tf.reduce_mean(tf.reshape(self.s_fake_B,[self.b_chan, self.imsize, -1]),axis=2),[1, -1])
            
            # Define generator loss functions
            self.lambda_c    = tf.placeholder_with_default([self.lambda_c],[1],name="lambda_c")
            self.lambda_c    = self.lambda_c[0]
            self.lambda_h    = tf.placeholder_with_default([self.lambda_h],[1],name="lambda_h")
            self.lambda_h    = self.lambda_h[0]
            
            self.dis_real_A  = self.disA.create(self.A,False)
            self.dis_real_Ah = self.disA_His.create(self.m_A,False)
            self.dis_real_B  = self.disB.create(self.B,False)
            self.dis_real_Bh = self.disB_His.create(self.m_B,False)
            self.dis_fake_A  = self.disA.create(self.fake_A,True)
            self.dis_fake_Ah = self.disA_His.create(self.m_fake_A,True)
            self.dis_fake_B  = self.disB.create(self.fake_B,True)
            self.dis_fake_Bh = self.disB_His.create(self.m_fake_B,True)
            
            self.cyc_A       = self.genA.create(self.fake_B,True)
            self.cyc_B       = self.genB.create(self.fake_A,True)
            
            
            # Define cycle loss (eq. 2)
            self.loss_cyc_A  = tf.reduce_mean(tf.abs(self.cyc_A-self.A))
            self.loss_cyc_B  = tf.reduce_mean(tf.abs(self.cyc_B-self.B))
            
            self.loss_cyc    = self.loss_cyc_A + self.loss_cyc_B
            
            # Define discriminator losses (eq. 1)
            self.loss_dis_A  = (tf.reduce_mean(tf.square(self.dis_real_A)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_A)))*0.5 +\
                               (tf.reduce_mean(tf.square(self.dis_real_Ah)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_Ah)))*0.5*self.lambda_h
                                
                               
            self.loss_dis_B  = (tf.reduce_mean(tf.square(self.dis_real_B)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_B)))*0.5 +\
                               (tf.reduce_mean(tf.square(self.dis_real_Bh)) +\
                                tf.reduce_mean(tf.square(1-self.dis_fake_Bh)))*0.5*self.lambda_h
            
            self.loss_gen_A  = tf.reduce_mean(tf.square(self.dis_fake_A)) +\
                               self.lambda_h * tf.reduce_mean(tf.square(self.dis_fake_Ah)) +\
                               self.lambda_c * self.loss_cyc/2.
            self.loss_gen_B  = tf.reduce_mean(tf.square(self.dis_fake_B)) +\
                               self.lambda_h * tf.reduce_mean(tf.square(self.dis_fake_Bh)) +\
                               self.lambda_c * self.loss_cyc/2.
                
        # Create the different optimizer
        with self.graph.as_default():
            # Optimizer for Gen
            self.list_gen        = []
            for var in tf.trainable_variables():
                if 'gen' in str(var):
                    self.list_gen.append(var)
            optimizer_gen   = tf.train.AdamOptimizer(learning_rate=self.relative_lr*0.0002,beta1=0.5)
            self.opt_gen    = optimizer_gen.minimize(self.loss_gen_A+self.loss_gen_B,var_list=self.list_gen)
            
            # Optimizer for Dis
            self.list_dis      = []
            for var in tf.trainable_variables():
                if 'dis' in str(var):
                    self.list_dis.append(var)
            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.relative_lr*0.0002,beta1=0.5)
            self.opt_dis  = optimizer_dis.minimize(self.loss_dis_A + self.loss_dis_B,var_list=self.list_dis)
            
    def save(self,sess):
        """
        Save the model parameter in a ckpt file. The filename is as 
        follows:
        ./Models/<mod_name>.ckpt
        
        INPUT: sess         - The current running session
        """
        self.saver.save(sess,"./Models/" + self.mod_name + ".ckpt")
            
    def init(self,sess):
        """
        Init the model. If the model exists in a file, load the model. Otherwise, initalize the variables
        
        INPUT: sess         - The current running session
        """
        if not os.path.isfile(\
                "./Models/" + self.mod_name + ".ckpt.meta"):
            sess.run(tf.global_variables_initializer())
            return 0
        else:
            if self.gen_only:
                sess.run(tf.global_variables_initializer())
            self.load(sess)
            return 1
    
    def load(self,sess):
        """
        Load the model from the parameter file:
        ./Models/<mod_name>.ckpt
        
        INPUT: sess         - The current running session
        """
        self.saver.restore(sess, "./Models/" + self.mod_name + ".ckpt")
    
    def train(self,batch_size=32,lambda_c=0.,lambda_h=0.,epoch=0,save=True,syn_noise=0.,real_noise=0.):
        f              = h5py.File(self.data_file,"r")
        
        num_samples    = min(self.a_size,self.b_size)
        num_iterations = num_samples // batch_size
        
        a_order        = np.random.permutation(self.a_size)
        b_order        = np.random.permutation(self.b_size)
        
        if self.verbose:
            print('lambda_c: ' + str(lambda_c))
            print('lambda_h: ' + str(lambda_h))
        
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
            
            vec_lcA     = []
            vec_lcB     = []
            
            vec_ldrA    = []
            vec_ldrAh   = []
            vec_ldrB    = []
            vec_ldrBh   = []
            vec_ldfA    = []
            vec_ldfAh   = []
            vec_ldfB    = []
            vec_ldfBh   = []
            
            vec_l_dis_A = []
            vec_l_dis_B = []
            vec_l_gen_A = []
            vec_l_gen_B = []
            
            rel_lr = 1.
            if epoch > 100:
                rel_lr = 2. - epoch/100.
            
            if epoch < 100:
                rel_noise = 0.9**epoch
            else:
                rel_noise = 0.
            
            for iteration in range(num_iterations):    
                images_a   = f['A/data'][np.sort(a_order[(iteration*batch_size):((iteration+1)*batch_size)]),:,:,:]
                images_b   = f['B/data'][np.sort(b_order[(iteration*batch_size):((iteration+1)*batch_size)]),:,:,:]
                if images_a.dtype=='uint8':
                    images_a=images_a/float(2**8-1)
                elif images_a.dtype=='uint16':
                    images_a=images_a/float(2**16-1)
                else:
                    raise ValueError('Dataset A is not int8 or int16')
                if images_b.dtype=='uint8':
                    images_b=images_b/float(2**8-1)
                elif images_b.dtype=='uint16':
                    images_b=images_b/float(2**16-1)
                else:
                    raise ValueError('Dataset B is not int8 or int16')
                    
                images_a  += np.random.randn(*images_a.shape)*real_noise
                images_b  += np.random.randn(*images_b.shape)*syn_noise

                _, l_gen_A, im_fake_A, l_gen_B, im_fake_B, cyc_A, cyc_B, sA, sB, sfA, sfB, lcA, lcB = sess.run([self.opt_gen,\
                                                                        self.loss_gen_A,\
                                                                        self.fake_A,\
                                                                        self.loss_gen_B,\
                                                                        self.fake_B,\
                                                                        self.cyc_A,\
                                                                        self.cyc_B,\
                                                                        self.s_A,self.s_B,self.s_fake_A,self.s_fake_B,\
                                                                        self.loss_cyc_A,\
                                                                        self.loss_cyc_B],\
                                                feed_dict={self.A: images_a,\
                                                           self.B: images_b,\
                                                           self.lambda_c: lambda_c,\
                                                           self.lambda_h: lambda_h,\
                                                           self.relative_lr: rel_lr,\
                                                           self.rel_dis_noise: rel_noise})

                if self.temp_b_s >= self.buffer_size:
                    rand_vec_a = np.random.permutation(self.buffer_size)[:batch_size]
                    rand_vec_b = np.random.permutation(self.buffer_size)[:batch_size]
                    
                    self.buffer_real_a[rand_vec_a,...] = images_a
                    self.buffer_real_b[rand_vec_b,...] = images_b
                    self.buffer_fake_a[rand_vec_a,...] = im_fake_A
                    self.buffer_fake_b[rand_vec_b,...] = im_fake_B
                else:
                    low                                = int(self.temp_b_s)
                    high                               = int(min(self.temp_b_s + batch_size,self.buffer_size))
                    self.temp_b_s                      = high
                    
                    self.buffer_real_a[low:high,...]   = images_a[:(high-low),...]
                    self.buffer_real_b[low:high,...]   = images_b[:(high-low),...]
                    self.buffer_fake_a[low:high,...]   = im_fake_A[:(high-low),...]
                    self.buffer_fake_b[low:high,...]   = im_fake_B[:(high-low),...]
                    
                # Create dataset out of buffer and gen images to train dis
                dis_real_a                         = np.copy(images_a)
                dis_real_b                         = np.copy(images_b)
                dis_fake_a                         = np.copy(im_fake_A)
                dis_fake_b                         = np.copy(im_fake_B)
                    
                half_b_s                           = int(batch_size/2)
                rand_vec_a                         = np.random.permutation(self.temp_b_s)[:half_b_s]
                rand_vec_b                         = np.random.permutation(self.temp_b_s)[:half_b_s]
                dis_real_a[:half_b_s,...]          =  self.buffer_real_a[rand_vec_a,...]
                dis_fake_a[:half_b_s,...]          =  self.buffer_fake_a[rand_vec_a,...]
                dis_real_b[:half_b_s,...]          =  self.buffer_real_b[rand_vec_b,...]
                dis_fake_b[:half_b_s,...]          =  self.buffer_fake_b[rand_vec_b,...]
                                
                _, l_dis_A, l_dis_B, \
                ldrA,ldrAh,ldfA,ldfAh,\
                ldrB,ldrBh,ldfB,ldfBh = sess.run([\
                                                self.opt_dis,
                                                self.loss_dis_A,
                                                self.loss_dis_B,
                                                self.dis_real_A,
                                                self.dis_real_Ah,
                                                self.dis_fake_A,
                                                self.dis_fake_Ah,
                                                self.dis_real_B,
                                                self.dis_real_Bh,
                                                self.dis_fake_B,
                                                self.dis_fake_Bh],feed_dict={self.A: dis_real_a,\
                                                                             self.B: dis_real_b,\
                                                                             self.fake_A: dis_fake_a,\
                                                                             self.fake_B: dis_fake_b,\
                                                                             self.lambda_c: lambda_c,\
                                                                             self.lambda_h: lambda_h,\
                                                                             self.relative_lr: rel_lr,\
                                                                             self.rel_dis_noise: rel_noise})
                
                vec_l_dis_A.append(l_dis_A)
                vec_l_dis_B.append(l_dis_B)
                vec_l_gen_A.append(l_gen_A)
                vec_l_gen_B.append(l_gen_B)

                vec_lcA.append(lcA)
                vec_lcB.append(lcB)
                
                vec_ldrA.append(ldrA)
                vec_ldrAh.append(ldrAh)
                vec_ldrB.append(ldrB)
                vec_ldrBh.append(ldrBh)
                vec_ldfA.append(ldfA)
                vec_ldfAh.append(ldfAh)
                vec_ldfB.append(ldfB)
                vec_ldfBh.append(ldfBh)

                if np.shape(images_b)[-1]==4:

                    images_b=np.vstack((images_b[0,:,:,0:3],np.tile(images_b[0,:,:,3].reshape(320,320,1),[1,1,3])))
                    im_fake_B=np.vstack((im_fake_B[0,:,:,0:3],np.tile(im_fake_B[0,:,:,3].reshape(320,320,1),[1,1,3])))
                    cyc_B=np.vstack((cyc_B[0,:,:,0:3],np.tile(cyc_B[0,:,:,3].reshape(320,320,1),[1,1,3])))
                    images_b=images_b[np.newaxis,:,:,:]
                    im_fake_B=im_fake_B[np.newaxis,:,:,:]
                    cyc_B=cyc_B[np.newaxis,:,:,:]

                if iteration%5==0:
                    sneak_peak=Utilities.produce_tiled_images(images_a,images_b,im_fake_A, im_fake_B,cyc_A,cyc_B)
                        
                    cv2.imshow("",sneak_peak[:,:,[2,1,0]])
                    cv2.waitKey(1)
                    
                print("\rTrain: {}/{} ({:.1f}%)".format(iteration+1, num_iterations,(iteration) * 100 / (num_iterations-1)) + \
                      "          Loss_dis_A={:.4f},   Loss_dis_B={:.4f}".format(np.mean(vec_l_dis_A),np.mean(vec_l_dis_B)) + \
                      ",   Loss_gen_A={:.4f},   Loss_gen_B={:.4f}".format(np.mean(vec_l_gen_A),np.mean(vec_l_gen_B))\
                          ,end="        ")
            
            # Save model
            if save:
                self.save(sess)
                cv2.imwrite("./Models/Images/" + self.mod_name + "_Epoch_" + str(epoch) + ".png",sneak_peak[:,:,[2,1,0]]*255)
            print("")
        
        f.close()
        
        loss_gen_A = [np.mean(np.square(np.array(vec_ldfA))),np.mean(np.square(np.array(vec_ldfAh))),np.mean(np.array(lcA))]
        loss_gen_B = [np.mean(np.square(np.array(vec_ldfB))),np.mean(np.square(np.array(vec_ldfBh))),np.mean(np.array(lcB))]
        loss_dis_A = [np.mean(np.square(np.array(vec_ldrA))),np.mean(np.square(1.-np.array(vec_ldfA))),\
                      np.mean(np.square(np.array(vec_ldrAh))),np.mean(np.square(1.-np.array(vec_ldfAh)))]
        loss_dis_B = [np.mean(np.square(np.array(vec_ldrB))),np.mean(np.square(1.-np.array(vec_ldfB))),\
                      np.mean(np.square(np.array(vec_ldrBh))),np.mean(np.square(1.-np.array(vec_ldfBh)))]
        
        return [loss_gen_A,loss_gen_B,loss_dis_A,loss_dis_B]

    def predict(self,lambda_c=0.,lambda_h=0.):
        f              = h5py.File(self.data_file,"r")
        
        rand_a     = np.random.randint(self.a_size-32)
        rand_b     = np.random.randint(self.b_size-32)
        
        images_a   = f['A/data'][rand_a:(rand_a+32),:,:,:]/255.
        images_b   = f['B/data'][rand_b:(rand_b+32),:,:,:]/255.
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
                
            fake_A, fake_B, cyc_A, cyc_B = \
                sess.run([self.fake_A,self.fake_B,self.cyc_A,self.cyc_B],\
                         feed_dict={self.A: images_a,\
                                    self.B: images_b,\
                                    self.lambda_c: lambda_c,\
                                    self.lambda_h: lambda_h})
            
        f.close()
        return images_a, images_b, fake_A, fake_B, cyc_A, cyc_B
    
    def generator_A(self,batch_size=32,lambda_c=0.,lambda_h=0.):
        f              = h5py.File(self.data_file,"r")
        f_save         = h5py.File("./Models/" + self.mod_name + '_gen_A.h5',"w")
        
        # Find number of samples
        num_samples    = self.b_size
        num_iterations = num_samples // batch_size
                
        gen_data       = np.zeros((f['B/data'].shape[0],f['B/data'].shape[1],f['B/data'].shape[2],f['A/data'].shape[3]),dtype=np.uint16)
        
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
            
            for iteration in range(num_iterations):    
                images_b   = f['B/data'][(iteration*batch_size):((iteration+1)*batch_size),:,:,:]
                if images_b.dtype=='uint8':
                    images_b=images_b/float(2**8-1)
                elif images_b.dtype=='uint16':
                    images_b=images_b/float(2**16-1)
                else:
                    raise ValueError('Dataset B is not int8 or int16')

                gen_A = sess.run(self.fake_A,feed_dict={self.B: images_b,\
                                                        self.lambda_c: lambda_c,\
                                                        self.lambda_h: lambda_h})
                gen_data[(iteration*batch_size):((iteration+1)*batch_size),:,:,:] = (np.minimum(np.maximum(gen_A,0),1)*(2**16-1)).astype(np.uint16)
                
                print("\rGenerator A: {}/{} ({:.1f}%)".format(iteration+1, num_iterations, iteration*100/(num_iterations-1)),end="   ")
        
        group = f_save.create_group('A')
        group.create_dataset(name='data', data=gen_data,dtype=np.uint16)
        
        f_save.close()
        f.close()
        
        return None
    
    def generator_B(self,batch_size=32,lambda_c=0.,lambda_h=0.):
        f              = h5py.File(self.data_file,"r")
        f_save         = h5py.File("./Models/" + self.mod_name + '_gen_B.h5',"w")
        
        # Find number of samples
        num_samples    = self.a_size
        num_iterations = num_samples // batch_size
                
        gen_data       = np.zeros((f['A/data'].shape[0],f['A/data'].shape[1],f['A/data'].shape[2],f['B/data'].shape[3]),dtype=np.uint16)
        
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
            
            for iteration in range(num_iterations):    
                images_a   = f['A/data'][(iteration*batch_size):((iteration+1)*batch_size),:,:,:]
                if images_a.dtype=='uint8':
                    images_a=images_a/float(2**8-1)
                elif images_a.dtype=='uint16':
                    images_a=images_a/float(2**16-1)
                else:
                    raise ValueError('Dataset A is not int8 or int16')

                gen_B = sess.run(self.fake_B,feed_dict={self.A: images_a,\
                                                        self.lambda_c: lambda_c,\
                                                        self.lambda_h: lambda_h})
                gen_data[(iteration*batch_size):((iteration+1)*batch_size),:,:,:] = (np.minimum(np.maximum(gen_B,0),1)*(2**16-1)).astype(np.uint16)
                
                print("\rGenerator B: {}/{} ({:.1f}%)".format(iteration+1, num_iterations, iteration*100/(num_iterations-1)),end="   ")
        
        group = f_save.create_group('B')
        group.create_dataset(name='data', data=gen_data,dtype=np.uint16)
        
        f_save.close()
        f.close()
        
        return None
        
    
    def get_loss(self,lambda_c=0.,lambda_h=0.):
        f              = h5py.File(self.data_file,"r")
        
        rand_a     = np.random.randint(self.a_size-32)
        rand_b     = np.random.randint(self.b_size-32)
        
        images_a   = f['A/data'][rand_a:(rand_a+32),:,:,:]/255.
        images_b   = f['B/data'][rand_b:(rand_b+32),:,:,:]/255.
        with tf.Session(graph=self.graph) as sess:
            # initialize variables
            self.init(sess)
                
            l_rA,l_rB,l_fA,l_fB = \
                sess.run([self.dis_real_A,self.dis_real_B,self.dis_fake_A,self.dis_fake_B,],\
                         feed_dict={self.A: images_a,\
                                    self.B: images_b,\
                                    self.lambda_c: lambda_c,\
                                    self.lambda_h: lambda_h})
            
        f.close()
        return l_rA,l_rB,l_fA,l_fB
