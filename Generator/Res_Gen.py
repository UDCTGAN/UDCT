from __future__ import division, print_function, unicode_literals

import tensorflow as tf

class ResGen:
    """
    This class is creating a ResNet generator as described by Zhu et al. 2018. 
    -) save()      - Save the current model parameter
    -) create()    - Create the model layers (graph construction)
    -) init()      - Initialize the model (load model if exists)
    -) load()      - Load the parameters from the file
    -) run()       - ToDo write this
    
    Only the following functions should be called from outside:
    -) create()
    -) constructor
    """
    
    def __init__(self,
            gen_name,
            out_dim,
            gen_dim=    [32,   64,128,   128,128,128,128,128,128,   64,32     ],
            kernel_size=[7,    3,3,      3,3,3,3,3,3,               3,3,     7],
            deconv='transpose',
            verbose=False):
        
#            gen_dim=    [64,   128,256,   256,256,256,256,256,256,256,256,256,   128,64    ],
#            kernel_size=[7,    3,3,       3,3,3,3,3,3,3,3,3,                     3,3,     7]):
        """
        Create a generator model (init). It will check, if a model with such a name has already been saved. If so, the model 
        is being loaded. Otherwise, a new model with this name will be created. It will only be saved, if the save function 
        is being called. The describtion of every parameter is given in the code below.
        
        INPUT: gen_name      - This is the name of the generator. It is mainly used to establish the place, where the model 
                               is being saved.
               gen_dim       - The number of channels in every layer. The first two elements are stride-2 convolutional layers.
                               The last two layers (1 value) are fractional stride 1/2 convolutional layers. Everything in between 
                               is a  ResNet layer.
               
               kernel_size   - The kernel sizes of all layers. The dimension of this list must be the same as of gen_dim + 1.
                              
        OUTPUT:              - The model
        """
        self.gen_name         = gen_name
        self.out_dim          = out_dim
        self.gen_dim          = gen_dim
        self.kernel_size      = kernel_size
        self.deconv           = deconv
        self.verbose          = verbose
        
        if len(gen_dim) + 1 != len(kernel_size):
            raise NameError('The dimensions of the ResGenerator are wrong')
        
        
    
    def create(self,X,reuse=True):
        num_layers = len(self.kernel_size)
        
        layer_list = []
        layer_list.append(X)
        
        if self.verbose:
            print('-------------------------------')
            print(' ')
            print('Create generator ' + self.gen_name)
            print(' ')
            print('Number of layers: ' + str(num_layers))
            print('Input diminesion: ' + str(tf.shape(X)))
            print(' ')
        
        for i in range(num_layers):
            if (i < (num_layers-3)) or (i==(num_layers - 1)):
                ps = int((self.kernel_size[i]-1)/2) # pad size
                new_pad = tf.pad(layer_list[-1],[[0,0],[ps,ps],[ps,ps],[0,0]],"Reflect")
                if self.verbose:
                    print('- - - - - - - - - - - - - - - -')
                    print('Load last layer: Do padding with size ' + str(ps))
            else:
                new_pad = layer_list[-1]
                if self.verbose:
                    print('- - - - - - - - - - - - - - - -')
                    print('Load last layer: No padding')
            if i==0 or i==(num_layers - 1):
                if i==0:
                    filters = self.gen_dim[i]
                else:
                    filters = self.out_dim
                new_conv = tf.layers.conv2d(new_pad,
                                            filters=filters,
                                            kernel_size=self.kernel_size[i],
                                            strides=(1,1),
                                            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                            padding='valid',
                                            reuse=reuse,
                                            name='gen_'+self.gen_name+'_conv_'+str(i))
                if self.verbose:
                    print('Conv layer stride 1 with kernel size ' + str(self.kernel_size[i]) + ' and number of filters ' + str(filters))
            elif i < 3:
                # Conv layers
                new_conv = tf.layers.conv2d(new_pad,
                                            filters=self.gen_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            strides=(2,2),
                                            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                            padding='valid',
                                            reuse=reuse,
                                            name='gen_'+self.gen_name+'_conv_'+str(i))
                if self.verbose:
                    print('Conv layer stride 2 with kernel size ' + str(self.kernel_size[i]) + ' and number of filters ' + str(self.gen_dim[i]))
            elif i < num_layers-3: 
                # Res layers
                new_conv_0 = tf.layers.conv2d(new_pad,
                                            filters=self.gen_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            strides=(1,1),
                                            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                            padding='valid',
                                            reuse=reuse,
                                            name='gen_'+self.gen_name+'_conv0_'+str(i))
#                new_norm_0 = tf.contrib.layers.instance_norm(new_conv_0,reuse=reuse,scale=False,center=False,scope='gen_'+self.gen_name+'_bnorm0_'+str(i),trainable=False)
                new_mean_0, new_var_0 = tf.nn.moments(new_conv_0,axes=(1,2),keep_dims=True)
                new_norm_0 = (new_conv_0 - new_mean_0) / tf.sqrt(new_var_0 + 1e-5)
                new_layer_0= tf.nn.relu(new_norm_0)
                new_pad_0 = tf.pad(new_layer_0,[[0,0],[ps,ps],[ps,ps],[0,0]],"Reflect")
                
                new_conv = tf.layers.conv2d(new_pad_0,
                                            filters=self.gen_dim[i],
                                            kernel_size=self.kernel_size[i],
                                            strides=(1,1),
                                            kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                            padding='valid',
                                            reuse=reuse,
                                            name='gen_'+self.gen_name+'_conv_'+str(i))
                
                if self.verbose:
                    print('Conv layer stride 1 with kernel size ' + str(self.kernel_size[i]) + ' and number of filters ' + str(self.gen_dim[i]))
                    print('Instance Normalization')
                    print('Relu')
                    print('Padding with pad size of ' + str(ps))
                    print('Conv layer stride 1 with kernel size ' + str(self.kernel_size[i]) + ' and number of filters ' + str(self.gen_dim[i]))
            else:
                # Deconv layers
                if self.deconv == 'transpose':
                    new_conv = tf.layers.conv2d_transpose(new_pad,
                                                          filters=self.gen_dim[i],
                                                          kernel_size=self.kernel_size[i],
                                                          strides=(2,2),
                                                          kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                                          padding='same',
                                                          reuse=reuse,
                                                          name='gen_'+self.gen_name+'_conv_'+str(i))
                    
                    if self.verbose:
                        print('Conv transpose layer stride 2 with kernel size ' + str(self.kernel_size[i]) + ' and number of filters ' + str(self.gen_dim[i]))
                elif self.deconv == 'resize':
                    if i == num_layers-3:
                        new_resize = tf.image.resize_images(new_pad,[tf.cast(tf.shape(X)[1]/2,dtype=tf.int32),\
                                                                     tf.cast(tf.shape(X)[2]/2,dtype=tf.int32)])
    
                    else:
                        new_resize = tf.image.resize_images(new_pad,[tf.shape(X)[1],tf.shape(X)[2]])
                    ps = int((self.kernel_size[i]-1)/2) # pad size
                    new_pad_0 = tf.pad(new_resize,[[0,0],[ps,ps],[ps,ps],[0,0]],"Reflect")
                    new_conv = tf.layers.conv2d(new_pad_0,
                                                filters=self.gen_dim[i],
                                                kernel_size=self.kernel_size[i],
                                                strides=(1,1),
                                                kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                                                padding='valid',
                                                reuse=reuse,
                                                name='gen_'+self.gen_name+'_conv_'+str(i))
                    if self.verbose:
                        print('Resize image')
                        print('Padding with pad size of ' + str(ps))
                        print('Conv layer stride 1 with kernel size ' + str(self.kernel_size[i]) + ' and number of filters ' + str(self.gen_dim[i]))
                else:
                    print('Unknown deconvolution method')
            if i < num_layers - 1:
#                new_norm = tf.contrib.layers.instance_norm(new_conv,reuse=reuse,scale=False,center=False,\
#                                                           scope='gen_'+self.gen_name+'_bnorm_'+str(i),trainable=False)
                new_mean, new_var = tf.nn.moments(new_conv,axes=(1,2),keep_dims=True)
                new_norm = (new_conv - new_mean) / tf.sqrt(new_var + 1e-5)
                
                if self.verbose:
                    print('Instance normalization')
            else:
                new_norm = new_conv
            
            if i>=3 and i < num_layers-3:
                new_layer      = new_norm + layer_list[-1]
                if self.verbose:
                    print('Make residual layer (linear activation function)')
            elif i < num_layers-1:
                new_layer      = tf.nn.relu(new_norm)
                if self.verbose:
                    print('ReLu')
            else:
                self.bef_layer = new_norm
                new_layer      = (tf.nn.tanh(new_norm)+1)/2.
                if self.verbose:
                    print('[tanh(x)+1]/2')
            
            layer_list.append(new_layer)
            if self.verbose:
                print(' ')
                print('Final layer: ',new_layer)
                
        self.layer_list = layer_list
        return new_layer
