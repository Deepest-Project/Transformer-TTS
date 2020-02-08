from text import symbols


################################
# Experiment Parameters        #
################################
seed=1234
output_directory = 'training_log'
log_directory = 'waveglow_char_parallel'
data_path = '/media/disk1/lyh/LJSpeech-1.1/waveglow'
teacher_path = '/media/disk1/lyh/fastspeech'

training_files='filelists/ljs_audio_text_train_filelist.txt'
validation_files='filelists/ljs_audio_text_val_filelist.txt'
text_cleaners=['english_cleaners']


################################
# Audio Parameters             #
################################
sampling_rate=22050
filter_length=1024
hop_length=256
win_length=1024
n_mel_channels=80
mel_fmin=80.0 # waveglow:0
mel_fmax=7600.0 # waveglow:8000

################################
# Model Parameters             #
################################
n_symbols=len(symbols)
data_type='char_seq' # 'phone_seq'
symbols_embedding_dim=256
hidden_dim=256
dprenet_dim=256
postnet_dim=256
ff_dim=1024
n_heads=4
n_layers=6


################################
# Optimization Hyperparameters #
################################
lr=384**-0.5
warmup_steps=4000
grad_clip_thresh=1.0
n_bins=1
batch_size=16
accumulation=4
iters_per_plot=1000
iters_per_checkpoint=10000
train_steps = 200000