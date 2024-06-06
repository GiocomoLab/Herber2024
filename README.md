# Herber2024
 Code to replicate analyses in Herber et al. 2024

0. Download data from the Dryad repository (https://doi.org/10.5061/dryad.8cz8w9h0d) locally and unzip subfolders. Clone this repository.
1. Install & update conda. Ensure that Python version 3.8.16 is accessible to conda.
2. Create an environment matching the content of the aging MEC environment (conda create --name agingmec --file agingmec_env.txt python = 3.8.16)
3. In this environment, run all code in the notebook "Import & Filter MATLAB Data" adjusting paths to match your machine. This will generate a local filtered copy of 
the neural and behavioral data in Python accessible .npy files. This is a pre-requisite for any additional analysis and takes ~30-40 minutes to run 
for all sessions across both the Split Maze (SM) and Random Foraging (RF) tasks. 
4. Run the notebook "Shuffle Procedure" at least through the classification of putative inhibitory interneurons. After that, you have the option of 
using the output of the rest of the notebook (Dryad repo > shuffle_scores) or generating it yourself. If you would like to regenerate shuffle scores, note that the 
shuffle procedure required to identify functional cell types (e.g. spatial, grid, non-grid spatial, speed cells) is computationally intensive and/or time-
consuming. This is a necessary step before proceeding with neural analyses starting in Figure 2. You may find it helpful to run the shuffle using a cluster. 
5. To recapitulate results in individual figures except Figure 4, run code in the corresponding Python notebooks using the agingmec env, adjusting paths as needed. It is recommended
to run the figures in the following order: Figure 1/S1/S2, Figure 2/S3, Figure 3/S4, Figure S5, Figure 5, Figure 6, and then Figure S6. In particular, Figure 3/S4 and Figure S5 results depend on Figure 2/S3 outputs and Figure S5 results depend on Figure 3/S4 output.
6. To recapitulate results in Figure 4, create an environment matching the context of the LFP world environment (conda create --name LFPworld --file LFPworld_env.txt python = 3.8.16). 
Then, run the corresponding Jupyter notebook for Figure 4 within that environment.
    
