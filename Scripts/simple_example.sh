# Create the sample dataet
python create_h5_dataset.py ./Data/Example/Genuine/ ./Data/Example/Synthetic/ ./Data/Example/example_dataset.h5

# Create a Directory for model data
mkdir Models

# Train the network
python main.py --dataset=./Data/Example/example_dataset.h5 --name=example_model 
