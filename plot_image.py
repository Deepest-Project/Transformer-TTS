from text import *
import torch
import matplotlib.pyplot as plt


def plot_melspec(target, melspec, melspec_post, mel_lengths, text_lengths):
    fig, axes = plt.subplots(3,1,figsize=(20,30))
    L, T = text_lengths[-1], mel_lengths[-1]

    axes[0].imshow(target[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    axes[2].imshow(melspec_post[-1].detach().cpu()[:,:T],
                   origin='lower',
                   aspect='auto')

    return fig


def plot_alignments(alignments, text, mel_lengths, text_lengths, att_type):
    fig, axes = plt.subplots(6,4,figsize=(20,30))
    L, T = text_lengths[-1], mel_lengths[-1]

    for layer, head in [(0,0), (0,1), (0,2), (0,3),
                        (1,0), (1,1), (1,2), (1,3),
                        (2,0), (2,1), (2,2), (2,3),
                        (3,0), (3,1), (3,2), (3,3),
                        (4,0), (4,1), (4,2), (4,3),
                        (5,0), (5,1), (5,2), (5,3)]:
        
        if att_type=='enc':
            align = alignments[-1, layer, head].contiguous().detach().cpu()
            axes[layer,head].imshow(align[:L, :L], aspect='auto')
            axes[layer,head].xaxis.tick_top()
        
        elif att_type=='dec':
            align = alignments[-1, layer, head].contiguous().detach().cpu()
            axes[layer,head].imshow(align[:T, :T], aspect='auto')
            axes[layer,head].xaxis.tick_top()
        
        elif att_type=='enc_dec':
            align = alignments[-1, layer, head].transpose(0,1).contiguous().detach().cpu()
            axes[layer,head].imshow(align[:L, :T], origin='lower', aspect='auto')

        '''
        plt.xticks(range(T), [ f'{i}' if (i%10==0 or i==T-1) else ''
               for i in range(T) ])
        
        plt.yticks(range(L),
               sequence_to_text(text[-1].detach().cpu().numpy()[:L]))
        '''
        
    return fig

def plot_gate(gate_out):
    fig = plt.figure(figsize=(10,5))
    plt.plot(torch.sigmoid(gate_out[-1]).detach().cpu())
    return fig