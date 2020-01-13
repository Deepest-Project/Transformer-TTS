import torch
import torch.nn as nn


class TransformerLoss(nn.Module):
    def __init__(self):
        super(TransformerLoss, self).__init__()
        
    def forward(self, pred, target, guide):
        mel_out, mel_out_post = pred
        mel_target = target
        
        mel_loss = nn.L1Loss()(mel_out, mel_target) + nn.L1Loss()(mel_out_post, mel_target)
        guide_loss = self.guide_loss(guide)
        
        return mel_loss + 10*guide_loss
    
    '''
    Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks
    with Guided Attention (https://arxiv.org/abs/1710.08969)
    '''
    def guide_loss(self, guide):
        alignments, text_lengths, mel_lengths = guide
        W = alignments[0].new_zeros(alignments[0].size())

        for i, (T, L) in enumerate(zip(mel_lengths, text_lengths)):
            mel_seq = torch.arange(T).to(torch.float32).unsqueeze(-1)/T
            text_seq = torch.arange(L).to(torch.float32).unsqueeze(0)/L
            x = torch.pow(mel_seq-text_seq, 2)
            W[i, :T, :L] += alignments[0].new_tensor(1-torch.exp(-12.5*x))
        
        return torch.mean( torch.sum(torch.stack(alignments)*W, dim=0) )