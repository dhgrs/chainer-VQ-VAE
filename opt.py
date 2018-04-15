# parameters of training
batchsize = 16
lr = 2e-4
ema_mu = 0.9999  # not supported now
update_encoder = True
trigger = (500000, 'iteration')
evaluate_interval = (1, 'epoch')
snapshot_interval = (1000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '../VCTK-Corpus'
dataset = 'VCTK'
# root = 'CMU_ARCTIC'
# dataset = 'ARCTIC'
# root = 'voice_statistics'
# dataset = 'vs'
data_format = 'wav'
sr = 16000
quantize = 256
top_db = 20
length = 7680

# parameters of VQ
d = 512
k = 128

# parameters of Decoder(WaveNet)
n_loop = 3
n_layer = 10
filter_size = 2
residual_channels = 32
dilated_channels = 32
skip_channels = 128
use_logistic = False
n_mixture = 30
log_scale_min = -40
embed_channels = 128
use_deconv = False
dropout_zero_rate = 0.

# parameters of losses
beta = 0.25

# parameters of generating
use_ema = True
