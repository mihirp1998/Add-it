# Add-it
Unoffical Implementation of Add-it: Training-Free Object Insertion in Images With Pretrained Diffusion Models

## Get started:

To implement this in flux i had to edit the diffusers package - the custom diffusers package can be found [here] (https://github.com/mihirp1998/diffusers_addit)



You can simply install the custom diffusers via this: 

``pip install 'git+https://github.com/mihirp1998/diffusers_addit'``

To edit run:

``python main.py``

vary the gamma hyperparamter to vary the source importance, reducing the gamma gives more importance to the source image while increasing gives more importance to the target prompt.
