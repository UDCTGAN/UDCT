#!/bin/sh

# Create the raw dataset
echo "Creating the raw dataset"
cd ../notebooks/
jupyter nbconvert --to notebook --execute make_raw_colored_live_neurons.ipynb --ExecutePreprocessor.timeout=1800

# Create the hdf5 file
cd ../
mkdir -p Models
echo "Creating the hdf5 dataset"
python create_h5_dataset.py ./Data/Neuron_Col_Live/Raw/ ./Data/Neuron_Col_Live/Synthetic/ ./Data/Neuron_Col_Live/colored_live_neuron_dataset.h5

# Train the network
echo "Training the network"
python main.py --dataset=./Data/Neuron_Col_Live/colored_live_neuron_dataset.h5 --name=live_colored_neuron_new

# Create the generated synthetic images
python main.py --dataset=./Data/Neuron_Col_Live/colored_live_neuron_dataset.h5 --name=live_colored_neuron_new --mode=gen_B 
