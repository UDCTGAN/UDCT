#!/bin/sh

# Create the synthetic images for both live_vs_dead and colored_live neuron datasets
cd ../notebooks/
echo "Creating the synthetic dataset"
jupyter nbconvert --to notebook --execute make_synthetic_livedead_neurons_and_colored.ipynb --ExecutePreprocessor.timeout=1800

# Create the raw dataset
echo "Creating the raw dataset"
mkdir -p ../Data/Neuron_Dead_Live/Raw
cd ../Data/Neuron_Dead_Live/Raw/
wget https://downloads.lbb.ethz.ch/Data/lbb_raw_neuron_images.h5
cd ../../../notebooks/
jupyter nbconvert --to notebook --execute make_raw_dead_live_neurons.ipynb --ExecutePreprocessor.timeout=1800

# Create the hdf5 file
cd ../
mkdir -p Models
echo "Creating the hdf5 dataset"
python create_h5_dataset.py ./Data/Neuron_Dead_Live/Raw/ ./Data/Neuron_Dead_Live/Synthetic/ ./Data/Neuron_Dead_Live/live_dead_neuron_dataset.h5

# Train the network
echo "Training the network"
python main.py --dataset=./Data/Neuron_Dead_Live/live_dead_neuron_dataset.h5 --name=live_dead_neuron_new

# Create the generated synthetic images
python main.py --dataset=./Data/Neuron_Dead_Live/live_dead_neuron_dataset.h5 --name=live_dead_neuron_new --mode=gen_B 
