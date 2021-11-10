# VSAPIKE
Sparse Vector Symbolic Architecture using Spiking Neural Networks

## Jupyter notebook set up

1. Create a conda env and install modules from requirments.txt
2. install jupyter into the environment
    1. conda activate EnvName
    2. conda install jupyter
    3. python -m ipykernel install --user --name=EnvName
    4. jupyter notebook (start jupyter)
    5. Navigate to the notebook
    6. When starting the notebook for the first time select your env as the kernel. 
       If you fail to do this jupyter will likely not be able to find the brian2 modules. 
       This can be fixed using the jupyter menu: kernel->change kernel.

