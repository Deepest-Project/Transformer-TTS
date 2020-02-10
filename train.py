import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Model
import hparams
import random
from text import *
from torch.utils.tensorboard import SummaryWriter
from loss import TransformerLoss
from plot_image import *
from utils import prepare_dataloaders, save_checkpoint, lr_scheduling


def validate(model, criterion, val_loader, iteration, writer):
    model.eval()
    with torch.no_grad():
        n_data, val_loss = 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            text_padded, text_lengths, mel_padded, mel_lengths, gate_padded = [
                x.cuda() for x in batch
            ]
            mel_out, mel_out_post, enc_alignments, dec_alignments, enc_dec_alignments, gate_out = model(text_padded,
                                                                                                        mel_padded,
                                                                                                        text_lengths,
                                                                                                        mel_lengths)

            mel_loss, bce_loss, guide_loss = criterion((mel_out, mel_out_post, gate_out),
                                                       (mel_padded, gate_padded),
                                                       (enc_dec_alignments, text_lengths, mel_lengths))
            loss = mel_loss+bce_loss+guide_loss
            val_loss += loss.item() * len(batch[0])

        val_loss /= n_data

    writer.add_scalar('val_mel_loss', mel_loss.item(),
                      global_step=iteration//hparams.accumulation)
    writer.add_scalar('val_bce_loss', bce_loss.item(),
                      global_step=iteration//hparams.accumulation)
    writer.add_scalar('val_guide_loss', guide_loss.item(),
                      global_step=iteration//hparams.accumulation)
    
    mel_fig = plot_melspec(mel_padded,
                           mel_out,
                           mel_out_post,
                           mel_lengths,
                           text_lengths)
    writer.add_figure('Validation melspec', mel_fig,
                      global_step=iteration//hparams.accumulation)

    enc_align_fig = plot_alignments(enc_alignments,
                                    text_padded,
                                    mel_lengths,
                                    text_lengths,
                                   'enc')
    writer.add_figure('Validation enc_alignments', enc_align_fig,
                      global_step=iteration//hparams.accumulation)

    dec_align_fig = plot_alignments(dec_alignments,
                                    text_padded,
                                    mel_lengths,
                                    text_lengths,
                                   'dec')
    writer.add_figure('Validation dec_alignments', dec_align_fig,
                      global_step=iteration//hparams.accumulation)

    enc_dec_align_fig = plot_alignments(enc_dec_alignments,
                                        text_padded,
                                        mel_lengths,
                                        text_lengths,
                                       'enc_dec')
    writer.add_figure('Validation enc_dec_alignments', enc_dec_align_fig,
                      global_step=iteration//hparams.accumulation)
    
    gate_fig = plot_gate(gate_out)
    writer.add_figure('Validation gate_out', gate_fig,
                      global_step=iteration//hparams.accumulation)
    
    model.train()
    
def main():
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams)

    model = Model(hparams).cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams.lr,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    criterion = TransformerLoss()

    logging_path=f'{hparams.output_directory}/{hparams.log_directory}'
    if os.path.exists(logging_path):
        raise Exception('The experiment already exists')
    else:
        os.mkdir(logging_path)
        writer = SummaryWriter(logging_path)

    iteration, loss = 0, 0
    model.train()
    print("Training Start!!!")
    while iteration < (hparams.train_steps*hparams.accumulation):
        for i, batch in enumerate(train_loader):
            text_padded, text_lengths, mel_padded, mel_lengths, gate_padded = [
                x.cuda() for x in batch
            ]
            mel_out, mel_out_post, enc_alignments, dec_alignments, enc_dec_alignments, gate_out = model(text_padded,
                                                                                                        mel_padded,
                                                                                                        text_lengths,
                                                                                                        mel_lengths)
            mel_loss, bce_loss, guide_loss = criterion((mel_out, mel_out_post, gate_out),
                                                       (mel_padded, gate_padded),
                                                       (enc_dec_alignments, text_lengths, mel_lengths))

            sub_loss = (mel_loss+bce_loss+guide_loss)/hparams.accumulation
            sub_loss.backward()
            loss = loss + sub_loss.item()

            iteration += 1
            if iteration%hparams.accumulation == 0:
                lr_scheduling(optimizer, iteration//hparams.accumulation)
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                model.zero_grad()
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                  global_step=iteration//hparams.accumulation)
                writer.add_scalar('mel_loss', mel_loss.item(),
                                  global_step=iteration//hparams.accumulation)
                writer.add_scalar('bce_loss', bce_loss.item(),
                                  global_step=iteration//hparams.accumulation)
                writer.add_scalar('guide_loss', guide_loss.item(),
                                  global_step=iteration//hparams.accumulation)
                loss=0


            if iteration%(hparams.iters_per_plot*hparams.accumulation)==0:
                mel_fig = plot_melspec(mel_padded,
                                       mel_out,
                                       mel_out_post,
                                       mel_lengths,
                                       text_lengths)
                writer.add_figure('Train melspec', mel_fig,
                                  global_step=iteration//hparams.accumulation)

                enc_align_fig = plot_alignments(enc_alignments,
                                                text_padded,
                                                mel_lengths,
                                                text_lengths,
                                               'enc')
                writer.add_figure('Train enc_alignments', enc_align_fig,
                                  global_step=iteration//hparams.accumulation)

                dec_align_fig = plot_alignments(dec_alignments,
                                                text_padded,
                                                mel_lengths,
                                                text_lengths,
                                               'dec')
                writer.add_figure('Train dec_alignments', dec_align_fig,
                                  global_step=iteration//hparams.accumulation)

                enc_dec_align_fig = plot_alignments(enc_dec_alignments,
                                                    text_padded,
                                                    mel_lengths,
                                                    text_lengths,
                                                   'enc_dec')
                writer.add_figure('Train enc_dec_alignments', enc_dec_align_fig,
                                  global_step=iteration//hparams.accumulation)
                
                gate_fig = plot_gate(gate_out)
                writer.add_figure('Train gate_out', gate_fig,
                                  global_step=iteration//hparams.accumulation)


            if iteration%(hparams.iters_per_checkpoint*hparams.accumulation)==0:
                save_checkpoint(model,
                                optimizer,
                                hparams.lr,
                                iteration//hparams.accumulation,
                                filepath=logging_path)


            if iteration==(hparams.train_steps*hparams.accumulation):
                break

        validate(model, criterion, val_loader, iteration, writer)

    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    main()