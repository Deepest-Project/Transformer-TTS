import hparams
from torch.utils.data import DataLoader
from data_utils import TextMelLoader, TextMelCollate
import torch
from text import *
import matplotlib.pyplot as plt


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate()

    train_loader = DataLoader(trainset,
                              num_workers=3,
                              shuffle=True,
                              batch_size=hparams.batch_size, 
                              drop_last=True, 
                              collate_fn=collate_fn)
    
    val_loader = DataLoader(valset,
                            num_workers=1,
                            batch_size=hparams.batch_size,
                            collate_fn=collate_fn)
    
    return train_loader, val_loader, collate_fn


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')

    
def lr_scheduling(opt, step, init_lr=hparams.lr, warmup_steps=hparams.warmup_steps):
    opt.param_groups[0]['lr'] = init_lr * min(step ** -0.5, step * warmup_steps ** -1.5)
    return