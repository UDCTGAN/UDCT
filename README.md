# UDCT
You can find here the Cycle-GAN network (Tensorflow, Python 2.7+) with our histogram loss. Additionally, we provide the scripts we used to generate the synthetic datasets and the post-processings scripts. 

Our results can be found at https://www.biorxiv.org/content/biorxiv/early/2019/03/01/563734.full.pdf

## How to use
1. Clone or download the repository
2. Create a synthetic dataset similar to your real dataset or use example dataset in ./Data/Example
3. Exectute: <pre>python create_h5_dataset.py &lt;directory_of_raw_images&gt; &lt;directory_of_syn_images&gt; &lt;filename_of_hdf5_file&gt;  </pre><br />
   Example: <pre>python create_h5_dataset.py ./Data/Example/Genuine/ \\<br />./Data/Example/Synthetic/ ./Data/Example/example_dataset.h5</pre>
4. Execute: <pre> python main.py --dataset='./Data/..../dataset.h5' </pre>
5. This will create a network that is saved in ./Models along with a parameter textfile. To evaluate the results, launch visualize_results.ipynb
### Parameters:
