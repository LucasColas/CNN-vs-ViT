# CNN-vs-ViT

This code contains a comparison between Resnet (trained from scratch) and fine-tuned Vit (Vit and CNN+ViT) on MedMNIST.

You will need to install libraries.
```bash
pip install -r requirements.txt
```

Use of CUDA is encouraged.

## Files / notebooks 

### Data Augmentation 

data_augmentation.py is a script containing all the functions and classes needed for the data augmentation.

### Resnet 
resnet.py contains the code to create a resnet. training_resnet.py allows you to get the best model after a grid search. The best model is saved as a pth file.

### Vit and Hybrid-ViT
The ViT models experiments are done in their completed notebook which displays the metrics obtained. In the case of the ViT, the metrics are stored in the grid-search folder.
