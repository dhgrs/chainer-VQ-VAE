# parameters of training
batchsize = 4
lr = 2e-4
ema_mu = 0.9999  # not supported now
trigger = (500000, 'iteration')
evaluate_interval = (1, 'epoch')
snapshot_interval = (1000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '/data/datasets/VCTK-Corpus'
dataset_type = 'VCTK'
split_seed = 71

# parameters of preprocessing
sr = 16000
top_db = 20
input_dim = 256
quantize = 256
length = 7680
use_logistic = False

# parameters of VQ
d = 512
k = 128

# parameters of Decoder(WaveNet)
n_loop = 3
n_layer = 10
filter_size = 2
input_dim = input_dim
residual_channels = 512
dilated_channels = 512
skip_channels = 256
# quantize = quantize
# use_logistic = use_logistic
n_mixture = 10 * 3
log_scale_min = -40
global_condition_dim = 128
local_condition_dim = 128
dropout_zero_rate = 0

# parameters of losses
beta = 0.25

# parameters of generating
use_ema = True
apply_dropout = False
