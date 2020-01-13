from text import *
import matplotlib.pyplot as plt


def plot_image(target, melspec, melspec_post, alignments, text, mel_lengths, text_lengths):
    fig, axes = plt.subplots(3,3,figsize=(30,30))
    L, T = text_lengths[-1], mel_lengths[-1]

    axes[0,0].imshow(target[-1].detach().cpu()[:,:T],
                     origin='lower',
                     aspect='auto')

    axes[0,1].imshow(melspec[-1].detach().cpu()[:,:T],
                     origin='lower',
                     aspect='auto')

    axes[0,2].imshow(melspec_post[-1].detach().cpu()[:,:T],
                     origin='lower',
                     aspect='auto')

    for i, (r,c) in enumerate([(1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]):
        align = alignments[i][-1].transpose(0,1).contiguous().detach().cpu()
        axes[r,c].imshow(align[:L, :T],
                         origin='lower',
                         aspect='auto')

        plt.xticks(range(T), [ f'{i}' if (i%10==0 or i==T-1) else ''
               for i in range(T) ])
        
        plt.yticks(range(L),
               sequence_to_text(text[-1].detach().cpu().numpy()[:L]))
        
    return fig