import cycleGAN
import re
import sys
from os import environ as cuda_environment
import os
import numpy as np

if __name__ == "__main__":
    # List of floats
    sub_value_f = {}
    sub_value_f['lambda_c']    = 10.             # Loss multiplier for cycle
    sub_value_f['lambda_h']    = 1.              # Loss multiplier for histogram
    sub_value_f['dis_noise']   = 0.1             # Std of gauss noise added to Dis
    sub_value_f['syn_noise']   = 0.              # Add gaussian noise to syn images to make non-flat backgrounds 
    sub_value_f['real_noise']  = 0.              # Add gaussian noise to real images to make non-flat backgrounds 
    
    # List of ints
    sub_value_i = {}
    sub_value_i['epoch']       = 200             # Number of epochs to be trained
    sub_value_i['batch_size']  = 4               # Batch size for training
    sub_value_i['buffer_size'] = 50              # Number of history elements used for Dis
    sub_value_i['save']        = 1               # If not 0, model is saved
    sub_value_i['gpu']         = 0               # Choose the GPU ID (if only CPU training, choose nonexistent number)
    sub_value_i['verbose']     = 0               # If not 0, some network information is being plotted
    
    # List of strings
    sub_string = {}
    sub_string['name']         = 'unnamed'       # Name of model (should be unique). Is used to save/load models
    sub_string['dataset']      = 'pathtodata.h5' # Describes which h5 file is used
    sub_string['architecture'] = 'Res6'          # Network architecture: 'Res6' or 'Res9'
    sub_string['deconv']       = 'transpose'     # Upsampling method: 'transpose' or 'resize'
    sub_string['PatchGAN']     = 'Patch70'       # Choose the Gan type: 'Patch34', 'Patch70', 'Patch142', 'MultiPatch'
    sub_string['mode']         = 'training'      # 'train', 'gen_A', 'gen_B'
    
    # Create complete dictonary
    var_dict  = sub_string.copy()
    var_dict.update(sub_value_i)
    var_dict.update(sub_value_f)
    
    # Update all defined parameters in dictionary
    for arg_i in sys.argv[1:]:
        var   = re.search('(.*)\=', arg_i) # everything before the '='
        g_var = var.group(1)[2:]
        if g_var in sub_value_i:
            dtype = 'int'
        elif g_var in sub_value_f:
            dtype = 'float'
        elif g_var in sub_string:
            dtype = 'string'
        else:
            print("Unknown key word: " + g_var)
            print("Write parameters as: <key word>=<value>")
            print("Example: 'python main.py buffer_size=32'")
            print("Possible key words: " + str(var_dict.keys()))
            continue
        
        content   = re.search('\=(.*)',arg_i) # everything after the '='
        g_content = content.group(1)
        if dtype == 'int':
            var_dict[g_var] = int(g_content)
        elif dtype == 'float':
            var_dict[g_var] = float(g_content)
        else:
            var_dict[g_var] = g_content
    if not os.path.isfile(var_dict['dataset']):
        raise ValueError('Dataset does not exist. Specify loation of an existing h5 file.')
    # Get the dataset filename
    
    
    # Restrict usage of GPUs
    cuda_environment["CUDA_VISIBLE_DEVICES"]=str(var_dict['gpu'])
    with open('Models/'+var_dict['name']+"_params.txt", "w") as myfile:
        for key in sorted(var_dict):
            myfile.write(key + "," + str(var_dict[key]) + "\n")
    
    # Find out, if whole network is needed or only the generators
    gen_only = False
    if 'gen' in var_dict['mode']:
        gen_only = True
    
    # Define the model
    model = cycleGAN.Model(\
        mod_name=var_dict['name'],\
        data_file=var_dict['dataset'],\
        buffer_size=var_dict['buffer_size'],\
        dis_noise=var_dict['dis_noise'],\
        architecture=var_dict['architecture'],\
        lambda_c=var_dict['lambda_c'],\
        lambda_h=var_dict['lambda_h'],\
        deconv=var_dict['deconv'],\
        patchgan=var_dict['PatchGAN'],\
        verbose=(var_dict['verbose']!=0),\
        gen_only=gen_only)
    
    # Plot parameter properties, if applicable
    if var_dict['verbose']:
        # Print the number of parameters
        model.print_count_variables()
        model.print_train_and_not_train_variables()
        
        # Create a graph file
        model.save_graph()
            
    elif var_dict['mode'] == 'training':
        # Train the model
        loss_gen_A = []
        loss_gen_B = []
        loss_dis_A = []
        loss_dis_B = []
                
        for i in range(var_dict['epoch']):
            print('')
            print('Epoch: ' + str(i+1))
            print('')
            lgA,lgB,ldA,ldB = \
            model.train(batch_size=var_dict['batch_size'],\
                        lambda_c=var_dict['lambda_c'],\
                        lambda_h=var_dict['lambda_h'],\
                        save=bool(var_dict['save']),\
                        epoch=i,\
                        syn_noise=var_dict['syn_noise'],\
                        real_noise=var_dict['real_noise'])
            loss_gen_A.append(lgA)
            loss_gen_B.append(lgB)
            loss_dis_A.append(ldA)
            loss_dis_B.append(ldB)
            np.save("./Models/" + var_dict['name'] + '_loss_gen_A.npy',np.array(loss_gen_A).T)
            np.save("./Models/" + var_dict['name'] + '_loss_gen_B.npy',np.array(loss_gen_B).T)
            np.save("./Models/" + var_dict['name'] + '_loss_dis_A.npy',np.array(loss_dis_A).T)
            np.save("./Models/" + var_dict['name'] + '_loss_dis_B.npy',np.array(loss_dis_B).T)
            
    elif var_dict['mode'] == 'gen_A':
        model.generator_A(batch_size=var_dict['batch_size'],\
                          lambda_c=var_dict['lambda_c'],\
                          lambda_h=var_dict['lambda_h'])
        
    elif var_dict['mode'] == 'gen_B':
        model.generator_B(batch_size=var_dict['batch_size'],\
                          lambda_c=var_dict['lambda_c'],\
                          lambda_h=var_dict['lambda_h'])
