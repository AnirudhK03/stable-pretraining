"""
HCL on CIFAR-10
===============

Hard negative contrastive learning (HCL, Robinson et al., ICLR 2021) on
CIFAR-10 with a ResNet-18, compared against a SimCLR baseline trained with the
exact same recipe. ``HCLLoss`` is a drop-in replacement for ``NTXEntLoss`` in
the composable :func:`stable_pretraining.forward.simclr` pipeline, so the only
difference between the runs below is the loss.

The recipe follows the official implementation (https://github.com/joshr17/HCL):
Adam (lr 1e-3, weight decay 1e-6, constant), temperature 0.5, batch size 256,
projector ``Linear(512, 512) -> BN -> ReLU -> Linear(512, 128)``. Representations
are evaluated during training with an online linear probe and a k-NN (k=200)
classifier on the CIFAR-10 test set.

Usage::

    # SimCLR baseline (the repo's NTXEntLoss)
    python examples/hcl_cifar10.py --loss simclr

    # HCL (official defaults)
    python examples/hcl_cifar10.py --loss hcl --beta 1.0 --tau-plus 0.1

    # Debiased contrastive learning (HCL with beta=0)
    python examples/hcl_cifar10.py --loss hcl --beta 0 --tau-plus 0.1

    # Resume an interrupted run
    python examples/hcl_cifar10.py --loss hcl --resume /abs/path/to/last.ckpt

Runs are logged by the built-in registry; browse them with ``spt web``.
"""

import argparse

import lightning as pl
import torch
import torchmetrics
from torch import nn

import stable_pretraining as spt
from stable_pretraining import forward
from stable_pretraining.data import transforms
from stable_pretraining.methods.hcl import HCLLoss


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--loss", choices=["simclr", "hcl"], default="hcl")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau-plus", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def build_data(batch_size: int, num_workers: int) -> spt.data.DataModule:
    """CIFAR-10 with the official HCL / SimCLR augmentations."""
    view = transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )
    train_transform = transforms.MultiViewTransform([view, view])
    val_transform = transforms.Compose(
        transforms.RGB(),
        transforms.ToImage(**spt.data.static.CIFAR10),
    )

    train_dataset = spt.data.HFDataset(
        path="uoft-cs/cifar10",
        split="train",
        transform=train_transform,
        rename_columns={"img": "image"},
    )
    val_dataset = spt.data.HFDataset(
        path="uoft-cs/cifar10",
        split="test",
        transform=val_transform,
        rename_columns={"img": "image"},
    )

    # persistent_workers avoids re-spawning workers every epoch, which is slow on
    # macOS where DataLoader workers use spawn instead of fork.
    loader_kwargs = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, **loader_kwargs
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


def main():
    """Train SimCLR or HCL on CIFAR-10 and evaluate with linear and k-NN probes."""
    args = parse_args()

    backbone = spt.backbone.from_torchvision("resnet18", low_resolution=True)
    backbone.fc = nn.Identity()

    projector = nn.Sequential(
        nn.Linear(512, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 128),
    )

    if args.loss == "hcl":
        loss = HCLLoss(
            temperature=args.temperature, beta=args.beta, tau_plus=args.tau_plus
        )
    else:
        loss = spt.losses.NTXEntLoss(temperature=args.temperature)

    module = spt.Module(
        backbone=backbone,
        projector=projector,
        forward=forward.simclr,
        simclr_loss=loss,
        optim={
            "optimizer": {"type": "Adam", "lr": args.lr, "weight_decay": 1e-6},
            "scheduler": {"type": "ConstantLR", "factor": 1.0, "total_iters": 0},
            "interval": "epoch",
        },
    )

    linear_probe = spt.callbacks.OnlineProbe(
        module,
        name="linear_probe",
        input="embedding",
        target="label",
        probe=nn.Linear(512, 10),
        loss=nn.CrossEntropyLoss(),
        metrics={
            "top1": torchmetrics.classification.MulticlassAccuracy(10),
            "top5": torchmetrics.classification.MulticlassAccuracy(10, top_k=5),
        },
    )
    knn_probe = spt.callbacks.OnlineKNN(
        name="knn_probe",
        input="embedding",
        target="label",
        queue_length=20000,
        metrics={"top1": torchmetrics.classification.MulticlassAccuracy(10)},
        input_dim=512,
        k=200,
        temperature=args.temperature,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "auto",
        devices=1,
        precision="32-true",
        num_sanity_val_steps=0,
        callbacks=[linear_probe, knn_probe],
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=build_data(args.batch_size, args.num_workers),
        seed=args.seed,
        ckpt_path=args.resume,
        weights_only=args.resume is None,
    )
    manager()


if __name__ == "__main__":
    main()
