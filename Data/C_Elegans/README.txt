Disclaimer: 'We used the C.elegans infection live/dead image set version 1 provided by Fred Ausubel and available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].'
https://data.broadinstitute.org/bbbc/BBBC010/

For copyright reasons, the C. elegans dataset needs to be created by downloading the data from the Broad Bioimage Bechnmark Collection.

To do so, please follow the README.txt instructions in the following order:

1) README.txt in Original/

2) README.txt in Raw/

3) README.txt in Synthetic/

4) Execute in root directory: python create_h5_dataset.py ./Data/C_Elegans/Raw/ ./Data/C_Elegans/Synthetic/ ./Data/C_Elegans/c_elegans_dataset.h5

Instead, you can also execute the script 'train_UDCT_c_elegans.sh' in the directory ../Scripts. It does all these steps automatically.
