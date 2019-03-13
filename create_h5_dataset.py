#!/usr/bin/python

import numpy as np
import h5py
import cv2
import os
import sys

def get_file_list(data_path):
    """
    This function returns the list of all png images in a given directory, such as their dimensions. It returns an error, if the image dimensions are not consistent.
    
    Arguments:
        data_path      (string) Path to directory of the set of images to be extracted
        
    Returns:
        file_list      (List of strings) Filenames of all png images in dataset
        dimensions     (3 x integer) Dimensions of images: height, width, channels
        flag           (boolean) Is true, iff the images are grayscale
    """
    
    # Create list of all png files
    file_list = []
    for element in os.listdir(data_path):
        if element[-4:] == ".png":
            file_list.append(data_path + element)
    
    
    # Compare sizes
    dimensions = cv2.imread(file_list[0],cv2.IMREAD_UNCHANGED).shape
    for i in range(1,len(file_list)):
        if not np.array_equal(dimensions,cv2.imread(file_list[i],cv2.IMREAD_UNCHANGED).shape):
            raise Exception('The following two images have different dimensions. Please make sure all images in this directory have the same size \n\r ' +\
                            file_list[0] + '\n\r' +\
                            file_list[i])
    
    # Add a 3rd value two dimensions, if it does not exist (this means it has 1 data channel)
    flag = False
    if len(dimensions) == 2:
        dimensions = np.array([dimensions[0],dimensions[1],1])
        flag = True
            
    return file_list,dimensions,flag

def main():
    # Check if the right amount of arguments has been given to the program
    if len(sys.argv[1:]) != 3:
        print('This script recuires three arguments in order to work:')
        print('1: Path to directory containing the genuine/raw images (only png images in directory are used!)')
        print('2: Path to directory containing the synthetic images (only png images in directory are used!)')
        print('3: Output hdf5 filename')
        print(' ')
        print('Example: python create_h5_dataset.py ./Data/Example/Genuine/ ./Data/Example/Synthetic/ ./Data/Example/example_dataset.h5')
        print(' ')
        print('Script aborted')
        return -1
    
    # get the addresses
    raw_path    = sys.argv[1]
    syn_path    = sys.argv[2]
    filename    = sys.argv[3]
    
    # Create the output hdf5 file
    f           = h5py.File(filename,"w")
    
    # Save the raw dataset into the file
    raw_files, raw_dimensions, raw_flag = get_file_list(raw_path)
    
    num_samples = len(raw_files)
    num_channel = raw_dimensions[2]
    
    group       = f.create_group('A')
    group.create_dataset(name='num_samples', data=num_samples)
    group.create_dataset(name='num_channel', data=num_channel)
    dtype       = np.uint8
    
    data_A      = np.zeros([num_samples,\
                            raw_dimensions[0],\
                            raw_dimensions[1],\
                            num_channel], dtype=dtype)
    
    for idx,fname in enumerate(raw_files):
        if raw_flag: # This means, the images are gray scale
            data_A[idx,:,:,0] = np.array(cv2.imread(fname,cv2.IMREAD_GRAYSCALE))
        else:
            data_A[idx,:,:,:] = np.flip(np.array(cv2.imread(fname,cv2.IMREAD_COLOR)),2)
        
    print('Genuine dataset: ', group.create_dataset(name='data', data=(data_A),dtype=dtype))
    
    
    
    
    
    # Save the syn dataset into the file
    syn_files, syn_dimensions, syn_flag = get_file_list(syn_path)
    
    num_samples = len(syn_files)
    num_channel = syn_dimensions[2]
    
    group       = f.create_group('B')
    group.create_dataset(name='num_samples', data=num_samples)
    group.create_dataset(name='num_channel', data=num_channel)
    dtype       = np.uint8
    
    data_B      = np.zeros([num_samples,\
                            syn_dimensions[0],\
                            syn_dimensions[1],\
                            num_channel], dtype=dtype)
    
    for idx,fname in enumerate(syn_files):
        if syn_flag: # This means, the images are gray scale
            data_B[idx,:,:,0] = np.array(cv2.imread(fname,cv2.IMREAD_GRAYSCALE))
        else:
            data_B[idx,:,:,:] = np.flip(np.array(cv2.imread(fname,cv2.IMREAD_COLOR)),2)
        
    print('Synthetic dataset: ', group.create_dataset(name='data', data=(data_B),dtype=dtype))
    
    
    # Close the file
    f.close()
    
if __name__ == "__main__":
    main()
