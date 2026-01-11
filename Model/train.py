from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
import lightning as pl
from torch.utils.data import DataLoader
import torch
from Model.polyencoder import LightningModelWrapper
from Model.dataset import CreativityScoringDataset


def main():

    batch = 2
    epochs = 8
    devices = torch.cuda.device_count()
    pl.seed_everything(42)

    # Tokenizer you want to use
    # Ex. roberta-base
    tokenizer = "desired-tokenizer"

    trainDataset = CreativityScoringDataset("path/to/train.csv", tokenizer)
    valDataset = CreativityScoringDataset("path/to/val.csv", tokenizer)

    train_loader = DataLoader(trainDataset, batch_size=batch, shuffle=True, num_workers=15)
    val_loader = DataLoader(valDataset, batch_size=batch, shuffle=False, num_workers=15)
    wandb_logger = WandbLogger(project="project-name", name="run-name")

    model = LightningModelWrapper(tokenizer, wandb_logger, poly_m=256)

    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # Unfreeze n layers
    n = 0
    for layer in model.model.encoder.encoder.layer[-n:]:
        for param in layer.parameters():
            param.requires_grad = True

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=devices,
        precision="16",
        logger=wandb_logger,
        log_every_n_steps=10,
        accumulate_grad_batches=4,
        strategy=DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=0.8,
        val_check_interval=0.20,
    )
    trainer.fit(model, train_loader, val_loader)

    best_model = model

    testPath = "path/to/test.csv"



if __name__ == "__main__":
    main()
