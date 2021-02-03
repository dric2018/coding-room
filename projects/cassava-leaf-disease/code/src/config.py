import os


class Config:
    data_dir = os.path.abspath('../data/input/')
    models_dir = os.path.abspath('../models')
    logs_dir = os.path.abspath('../logs')
    train_data_dir = os.path.abspath('../data/input/train_images')
    test_data_dir = os.path.abspath('../data/input/test_images')
    num_epochs = 15
    lr = 2e-2
    resize = 500
    img_h = 400
    img_w = 400
    weight_decay = .01
    eps = 1e-8
    train_batch_size = 16
    test_batch_size = 16
    base_model = 'resnet34'
    seed_val = 2021
    num_workers = 2