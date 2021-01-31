import os


class Config:
    data_dir = os.path.abspath('../data')
    models_dir = os.path.abspath('../models')
    logs_dir = os.path.abspath('../logs')
    num_epochs = 2
    lr = 1e-2
    weight_decay = .01
    eps = 1e-8
    train_batch_size = 1024
    test_batch_size = 512
    base_model = None
