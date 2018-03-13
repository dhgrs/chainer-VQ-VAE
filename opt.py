# parameters of training
batchsize = 1
lr = 2e-4
update_encoder = True
trigger = (200000, 'iteration')
evaluate_interval = (1, 'epoch')
snapshot_interval = (1000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = 'VCTK-Corpus'
dataset = 'VCTK'
# root = 'CMU_ARCTIC'
# dataset = 'ARCTIC'
# root = 'voice_statistics'
# dataset = 'vs'
data_format = 'wav'
sr = 16000
mu = 256
top_db = 20
length = 7680

# parameters of VQ
d = 512
k = 128

# parameters of Decoder(WaveNet)
n_loop = 3
n_layer = 10
n_filter = 2
residual_channels = 64
dilated_channels = 64
skip_channels = 256
embed_channels = 128

# parameters of losses
beta = 0.25
