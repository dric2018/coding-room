# IMPPORTS
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from .config import Config

# CONSTANTS DEFINITION


class Model(pl.LightningModule):
    def __init__(self, config: dict):
        super(Model, self).__init__()
        try:
            self.save_hyperparameters(config)
        except:
            pass

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3))
        self.conv2 = torch.nn.Conv2d(32, 64, (3, 3))
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 128, (3, 3))
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.bn2 = torch.nn.BatchNorm2d(128)

        self.classifier = torch.nn.Linear(128 * 5 * 5, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.pool1(self.bn1(x)))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.pool2(self.bn2(x)))
        x = self.classifier(x.view(-1, 128 * 5 * 5))

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        acc = accuracy(preds.cpu(), y.cpu())

        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)

        return {'loss': loss,
                'accuracy': acc,
                "predictions": preds,
                'targets': y
                }

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.nn.functional.softmax(logits, dim=1)
        val_loss = torch.nn.CrossEntropyLoss()(logits, y)
        val_acc = accuracy(preds.cpu(), y.cpu())

        self.log('val_loss', val_loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log('val_acc', val_acc, prog_bar=True,
                 on_step=False, on_epoch=True)

        return {'loss': val_loss,
                'accuracy': val_acc,
                "predictions": preds,
                'targets': y
                }

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            lr=self.hparams.lr,
            params=self.parameters(),
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )
        return opt
