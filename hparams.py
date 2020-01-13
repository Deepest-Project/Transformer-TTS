from text import symbols


################################
# Experiment Parameters        #
################################
seed=1234
output_directory = 'training_log'
log_directory = 'transformer-tts'
data_path = ''
teacher_path = ''

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
mel_fmin=0.0
mel_fmax=8000.0
ref_level_db=20
min_level_db=-80

################################
# Model Parameters             #
################################
n_symbols=len(symbols)
symbols_embedding_dim=384
hidden_dim=384
ff_dim=1536
n_heads=2
n_layers=6


################################
# Optimization Hyperparameters #
################################
lr=0.05 # ~384^-0.5 = 0.05
warmup_steps=4000
grad_clip_thresh=1.0
batch_size=[48, 36, 30, 26, 24, 21, 20, 18, 17, 16]
accumulation=2
iters_per_plot=1000
iters_per_checkpoint=10000
train_steps = 200000