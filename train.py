import argparse
from typing import Dict
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data.dataset import DSPDataModule, ALL_FEATURES, LABEL_COLS
from models.mmoe import MMoE
import torch.nn.functional as F

class MultiTaskModule(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = MMoE(
            num_features=hparams['num_features'],
            num_fields=len(ALL_FEATURES),
            embed_dim=hparams['embed_dim'],
            experts_hidden=hparams['experts_hidden'],
            num_experts=hparams['num_experts'],
            task_hidden=hparams['task_hidden'],
            tasks=LABEL_COLS,
        )
        # Uncertainty weights for dynamic loss balancing
        self.log_vars = torch.nn.Parameter(torch.zeros(len(LABEL_COLS)))
        self.pos_weights = torch.tensor(hparams['pos_weights'])

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, batch):
        x, y = batch
        outputs = self(x)
        losses = []
        for i, task in enumerate(LABEL_COLS):
            loss = F.binary_cross_entropy(outputs[task], y[:, i], weight=self.pos_weights[i].to(self.device))
            precision = torch.exp(-self.log_vars[i])
            losses.append(precision * loss + self.log_vars[i])
            self.log(f"{task}_loss", loss, prog_bar=True, on_epoch=True)
            self.log(f"{task}_pred_mean", outputs[task].mean(), prog_bar=False, on_epoch=True)
        return sum(losses)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--experts_hidden', type=int, default=64)
    parser.add_argument('--task_hidden', type=int, default=32)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Placeholder; in a real system we should compute total unique categories offline
    num_features = 1_000_000
    pos_weights = [50.0, 200.0, 100.0]  # Example; compute from data

    hparams = {
        'embed_dim': args.embed_dim,
        'experts_hidden': args.experts_hidden,
        'task_hidden': args.task_hidden,
        'num_experts': args.num_experts,
        'lr': args.lr,
        'num_features': num_features,
        'pos_weights': pos_weights,
    }

    dm = DSPDataModule(args.data_path, batch_size=args.batch_size)

    model = MultiTaskModule(hparams)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best')
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus else 'cpu',
        devices=args.gpus if args.gpus else None,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    main()