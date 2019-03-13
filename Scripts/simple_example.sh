#!/bin/sh

# Create the sample dataset
cd ..
python create_h5_dataset.py ./Data/Example/Genuine/ ./Data/Example/Synthetic/ ./Data/Example/example_dataset.h5

# Create a Directory for model data
mkdir -p Models

# Train the network
python main.py --dataset=./Data/Example/example_dataset.h5 --name=example_model 
