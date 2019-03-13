#!/bin/sh

# Download the original BBBC dataset and extract it
cd ../Data/C_Elegans/Original
rm ./00*.png
rm ./1649_1109_0003*.tif
wget https://data.broadinstitute.org/bbbc/BBBC010/BBBC010_v1_images.zip
unzip BBBC010_v1_images.zip -d ./
rm BBBC010_v1_images.zip
mv BBBC010_v1_images/* ./
rm -R BBBC010_v1_images

# Transform the images to pngs
cd ../../../notebooks/
jupyter nbconvert --to notebook --execute transform_c_elegans_tif_to_png.ipynb --ExecutePreprocessor.timeout=1800

# Create the raw dataset
echo "Creating the raw dataset"
jupyter nbconvert --to notebook --execute make_raw_c_elegans.ipynb --ExecutePreprocessor.timeout=1800

# Create the synthetic dataset
echo "Creating the synthetic dataset"
jupyter nbconvert --to notebook --execute make_synthetic_c_elegans.ipynb --ExecutePreprocessor.timeout=1800

# Create the hdf5 file
cd ../
mkdir -p Models
echo "Creating the hdf5 dataset"
python create_h5_dataset.py ./Data/C_Elegans/Raw/ ./Data/C_Elegans/Synthetic/ ./Data/C_Elegans/c_elegans_dataset.h5

# Train the network
echo "Training the network"
python main.py --dataset=./Data/C_Elegans/c_elegans_dataset.h5 --name=c_elegans_new

# Create the generated synthetic images
python main.py --dataset=./Data/C_Elegans/c_elegans_dataset.h5 --name=c_elegans_new --mode=gen_B 
