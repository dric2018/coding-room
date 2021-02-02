import os


class Config:
    data_dir = os.path.abspath('../data/input/')
    models_dir = os.path.abspath('../models')
    logs_dir = os.path.abspath('../logs')
    train_data_dir = os.path.abspath('../data/input/train_images')
    test_data_dir = os.path.abspath('../data/input/test_images')
    num_epochs = 5
    lr = 1e-2
    resize = 400
    img_h = 350
    img_w = 350
    weight_decay = .01
    eps = 1e-8
    train_batch_size = 32
    test_batch_size = 32
    base_model = 'resnet50'
    seed_val = 2021
