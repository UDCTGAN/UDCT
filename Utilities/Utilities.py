import numpy as np

def produce_tiled_images(im_A,im_B,fake_A,fake_B,cyc_A,cyc_B):

    list_of_images=[im_A,im_B,fake_A,fake_B,cyc_A,cyc_B]
    for i in range(6):
        if np.shape(list_of_images[i])[-1]==1:
            list_of_images[i]=np.tile(list_of_images[i],[1,1,1,3])  
        list_of_images[i]=np.pad(list_of_images[i][0,:,:,:], ((20,20),(20,20),(0,0)), mode='constant', constant_values=[0.5])
    im_A,im_B,fake_A,fake_B,cyc_A,cyc_B=list_of_images
    a=np.vstack( (im_A,im_B))
    b=np.vstack( (fake_B,fake_A))
    c=np.vstack( (cyc_A,cyc_B))
    return np.hstack((a,b,c))
