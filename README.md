# UDCT
You can find here the Cycle-GAN network (Tensorflow, Python 2.7+) with our histogram loss. Additionally, we provide the scripts we used to generate the synthetic datasets. 

Our results can be found at https://www.biorxiv.org/content/biorxiv/early/2019/03/01/563734.full.pdf

## How to use
1. Clone or download the repository
2. Create a synthetic dataset similar to your real dataset or use example dataset in ./Data/Example
3. Execute: <pre>python create_h5_dataset.py &lt;directory_of_raw_images&gt; &lt;directory_of_syn_images&gt; &lt;filename_of_hdf5_file&gt;  </pre>
   Example: <pre>python create_h5_dataset.py ./Data/Example/Genuine/ \\<br />./Data/Example/Synthetic/ ./Data/Example/example_dataset.h5</pre>
4. Create the directory 'Models' in the root directory
5. Execute: <pre> python main.py --dataset=./Data/..../dataset.h5 --name=name_of_model </pre>
   Example: <pre> python main.py --dataset=./Data/Example/example_dataset.h5 --name=example_model </pre>
6. This will create a network that is saved in ./Models/ along with a parameter textfile. Furthermore, the average loss terms for each epoch are saved in this directory.
7. To generate the results after training, use:
   <pre> python main.py --dataset=./Data/Example/example_dataset.h5 --name=example_model --mode=gen_B </pre>
   The generated synthetic images can be found in ./Models/&lt;name_of_model&gt;_gen_B.h5
   
### Parameters

All parameters are of the shape: --&lt;parameter_name&gt;=&lt;value&gt; <br />
Below is the list of all possible parameters that can be set. The standard value used if the parameter is not defined is given in brackets

<b>name ('unnamed')</b><br />
Name of the model. This value should be unique to not load/overwrite old models. Its value must be changed to ensure functionality!
<br />

<b>dataset ('pathtodata.h5')</b><br />
Describes which h5 files is used. Its value must be changed to ensure functionality!
<br />

<b>architecture ('Res6')</b><br />
The network architecture for the generators. Currently, you can choose between 'Res6' and 'Res9', which corresponds to 6 and 9 residual layers, respectively.
<br />

<b>deconv ('transpose')</b><br />
Upsampling method used in the generators. You can either choose transpose CNNs ('transpose') or image resizing ('resize').
<br />

<b>PatchGAN ('Patch70')</b><br />
Different PatchGAN Architectures: 'Patch34', 'Patch70', or 'Patch142'. A mixture of these is possible: 'MultiPatch' (experimental).
<br />

<b>mode ('training')</b><br />
Decides what should be done with the network. You can either train it ('training'), or create generated images: 'gen_A' for raw images from synthetic images and 'gen_B' for synthetic images from raw images.
<br />



<b>dataset ('pathtodata.h5')</b><br />
Describes which h5 files is used. Its value must be changed to ensure functionality!
<br />

<b>lambda_c (10.)</b><br />
The loss multiplier of the cycle consistency term used while training the generators.
<br />

<b>lambda_h (1.)</b><br />
The loss multiplier of the histogram discriminators. If the histogram should not be used, set this term to 0.
<br />

<b>dis_noise (0.1)</b><br />
To make the network more stable, we added noise to the input of the discriminators, which slowly decays over time. This value describes how high the std of the gaussian noise is, which is added to the inputs.
<br />

<b>syn_noise (0.)</b><br />
It is possible to add gaussian noise to the synthetic dataset. Default: not used.
<br />

<b>real_noise (0.)</b><br />
It is possible to add gaussian noise to the real dataset. Default: not used.
<br />

<b>epoch (200)</b><br />
Number of training epochs.
<br />

<b>batch_size (4)</b><br />
Batch size during training.
<br />

<b>buffer_size (50)</b><br />
Size of the buffer (history) saved to train the discriminators. This makes the network more stable.
<br />

<b>save (1)</b><br />
If value is not 0, the network progress is saved at the end of each epoch.
<br />

<b>gpu (0)</b><br />
If multiple GPUs exist, this parameter choses which GPU should be used. Only one GPU can currently be used.
<br />

<b>verbose (0)</b><br />
If value is not 0, the network is more verbose.
<br />
