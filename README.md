# Herber2024
 Code to replicate analyses in Herber et al. 2024

0. Download data from https://doi.org/10.5061/dryad.8cz8w9h0d locally and unzip subfolders. Clone this repository.
1. Install & update conda. Ensure that Python version 3.8.16 is accessible to conda.
2. Create an environment matching the content of the aging MEC environment (conda create --name agingmec --file agingmec_env.txt python = 3.8.16)
3. Run all code in the notebook "Import & Filter MATLAB Data" adjusting paths to match your machine. This will generate a local filtered copy of 
the neural and behavioral data in Python accessible .npy files. This is a pre-requisite for any additional analysis and takes ~30-40 minutes to run 
for all sessions across both the Split Maze (SM) and Random Foraging (RF) tasks. 
4. The shuffle procedure required to identify functional cell types (e.g. spatial, grid, non-grid spatial, speed cells) is computationally intensive 
and/or time-consuming. It is recommended to run the code in the notebook "Shuffle Procedure" before proceeding with neural analyses starting after 
Figure 1. You may find it helpful to run this using a computing cluster and batch scripting. 
5. To recapitulate results, run code in the corresponding Python notebooks, adjusting paths as needed.     