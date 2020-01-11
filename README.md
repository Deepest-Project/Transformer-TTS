# Transformer-TTS
Implementation of ["Neural Speech Synthesis with Transformer Network"](https://arxiv.org/abs/1809.08895)  
This is implemented for [FastSpeech](https://github.com/Deepest-Project/FastSpeech), so I use FFTblock as a encoder.

# Notice  
1. Unlike the original paper, I didn't use the stop token prediction
2. I use additional ["guided attention loss"](https://arxiv.org/pdf/1710.08969.pdf) with a coefficient '10'
3. For later use in fastspeech, I change return values of the "torch.nn.functional.multi_head_attention_forward()"
4.  

```python
#before
return attn_output, attn_output_weights.sum(dim=1) / num_heads  

#after  
return attn_output, attn_output_weights
```
            
# Reference
1.NVIDIA/tacotron2: https://github.com/NVIDIA/tacotron2
2.espnet: https://github.com/espnet/espnet
3.soobinseo/Transformer-TTS: https://github.com/soobinseo/Transformer-TTS
