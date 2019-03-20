#!/bin/sh

# Create the raw dataset
echo "Creating the raw dataset"
mkdir -p ../Data/Nanowire/Raw
cd ../Data/Nanowire/Raw/
wget https://downloads.lbb.ethz.ch/Data/lbb_raw_nanowire_images.h5
cd ../../../notebooks/
jupyter nbconvert --to notebook --execute make_raw_nanowire.ipynb --ExecutePreprocessor.timeout=1800

# Create the synthetic images
mkdir -p ../Data/Nanowire/Synthetic
echo "Creating the synthetic dataset"
jupyter nbconvert --to notebook --execute make_synthetic_wires.ipynb --ExecutePreprocessor.timeout=1800

# Create the hdf5 file
cd ../
mkdir -p Models
echo "Creating the hdf5 dataset"
python create_h5_dataset.py ./Data/Nanowire/Raw/ ./Data/Nanowire/Synthetic/ ./Data/Nanowire/nanowire_dataset.h5

# Train the network
echo "Training the network"
python main.py --dataset=./Data/Nanowire/nanowire_dataset.h5 --name=nanowire_new

# Create the generated synthetic images
python main.py --dataset=./Data/Nanowire/nanowire_dataset.h5 --name=nanowire_new --mode=gen_B 
