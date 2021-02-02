# IMPPORTS
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
import torchvision
import torch.nn.functional as F
from timm import create_model

from .config import Config

# CONSTANTS DEFINITION



class Model(pl.LightningModule):
    def __init__(self, config: dict, from_timm=True, 
                 from_th_vision=False):
        super(Model, self).__init__()
        try:
            self.save_hyperparameters(config)
        except:
            pass
        
        # get backbone
        if from_timm:
            
            self.encoder = create_model(model_name=self.hparams.base_model, pretrained=True)
        else:
            self.encoder = getattr(thvision.models, self.hparams.base_model)(pretrained=True)
        
        
        # create classification layer
        self.classifier = th.nn.Linear(in_features=1000, out_features=5)
        self.dropout = th.nn.Dropout(p=.35)

    def forward(self, x):
        x = self.encoder(x)
        self.dropout(x)
        out = self.classifier(x)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['img'], batch['targets']
        logits = self(x)
        preds = F.log_softmax(logits, dim=1)
        loss = th.nn.NLLLoss()(logits, y)
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
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Train",
                                          avg_acc,
                                          self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['targets']
        logits = self(x)
        preds = F.log_softmax(logits, dim=1)
        val_loss = th.nn.NLLLoss()(logits, y)
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
        avg_loss = th.stack([x['loss'] for x in outputs]).mean()
        # acc
        avg_acc = th.stack([x['accuracy'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Validation",
                                          avg_loss,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Accuracy/Validation",
                                          avg_acc,
                                          self.current_epoch)

    def configure_optimizers(self):
        opt = th.optim.AdamW(
            lr=self.hparams.lr,
            params=self.parameters(),
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode='min',
            factor=0.1,
            patience=4,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=True,
            )
        return {"optimizer": opt, 
                "lr_scheduler": scheduler, 
                "monitor": "val_acc"}

