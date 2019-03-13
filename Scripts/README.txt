In this directory you can find scripts that can be exectuted in order to reproduce the results of the publication.

simple_example.sh
A simple example that tests if the code is executing properly.

train_UDCT_c_elegans.sh
This script downloads the C. elegans dataset from the Broad Bioimage Bechnmark Collection, creates the Raw/Syn dataset and trains a network. Attention: This script is deleting some data in the Data/C_Elegans/Original directory.

train_UDCT_dead_live_neurons.sh
This script creates the dead vs alive dataset for the neurons. Afterwards, it trains a UDCT cycleGAN on the dataset.


