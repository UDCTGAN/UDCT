# UDCT
You can find here the Cycle-GAN network (Tensorflow, Python 2.7+) with our histogram loss. Additionally, we provide the scripts we used to generate the synthetic datasets and the post-processings scripts. 

Our results can be found at https://www.biorxiv.org/content/biorxiv/early/2019/03/01/563734.full.pdf

## How to use
### 1. Create a synthetic dataset similar to your real dataset
### 2. Go to ./notebooks/ and create the .h5 file with create_dataset_h5.ipynb
### 3. cd ../ and execute python main.py --datasets=./Data/..../h5file.h5
### 4. This will create a network that is saved in ./Models along with a parameter textfile. To evaluate the results, launch visualize_results.ipynb
### Parameters:
