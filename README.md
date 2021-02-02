# coding-room

A bunch of codes and experiments

# Project 1 : [Kaggle cassava leaf diseases classification](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)

## Experiments

This is a week map for all the experiments I do.

### **Day 1**:

- Pytorch-lightning setup
- Configure data module & Model
- Hyperparams tried:
- Training on : Nvidia GTX 1060 (6.1k MB VRAM)

  - num_epochs = 10
  - lr = 3e-4
  - resize = 400
  - img_h = 350
  - img_w = 350
  - weight_decay = .01
  - eps = 1e-8
  - train_batch_size = 32
  - test_batch_size = 32
  - base_model = 'resnet34'
  - seed_val = 2021
  - Adam Optimizer with weight decay
  - ReduceLrOnPlateau (monitoring val_accuracy)
  - Using auto mixed precision (fp16)

- Results (Local):
  - Validation loss : 0.65
  - Validation accuracy : 0.79
- Results (Kaggle):
  - Test accuracy : 0.838

### **Day 2**:

- Pytorch-lightning setup
- Configure data module & Model
- Hyperparams tried:
- Training on : Nvidia Tesla P100 (16k MB VRAM)

  - num_epochs = 25
  - lr = 2e-2
  - resize = 600
  - img_h = 512
  - img_w = 512
  - weight_decay = .01
  - eps = 1e-8
  - train_batch_size = 64
  - test_batch_size = 32
  - base_model = 'resnet34'
  - seed_val = 2021
  - Adam Optimizer with weight decay
  - ReduceLrOnPlateau (monitoring val_accuracy)
  - Using auto mixed precision (fp16)

- Results (Local):
  - Validation loss : NaN
  - Validation accuracy : NaN
- Results (Kaggle):
  - Test accuracy : NaN
