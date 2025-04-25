# CNN-VS-ViT

This code contains a comparison between Resnet (trained from scratch) and fine-tuned Vit (Vit and CNN+ViT).

You will need to install libraries.
```bash
pip install -r requirements.txt
```

Use of CUDA is encouraged.

## Files / notebooks 
resnet.py contains the code to create a resnet. training_resnet.py allows you to get the best model after a grid search. The best model is saved as a pth file.
