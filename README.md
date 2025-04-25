# CNN-VS-ViT

This code contains a comparison between Resnet (trained from scratch) and fine-tuned Vit (Vit and CNN+ViT).

You will need to install : 
```
medmnist
torch
torchvision
numpy
matplotlib
scikit-learn
transformers
wandb
evaluate
pillow
```

Use of CUDA is encouraged.

## Files / notebooks 
resnet.py contains the code to create a resnet. training_resnet.py allows you to get the best model after a grid search. The best model is saved as a pth file. ResNet.ipynb is a notebook that was used to do some experiments. It is no longer necessary to understand the code of this notebook. resnet.py and training_resnet.py contain the updated code.
