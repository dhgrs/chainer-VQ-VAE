# parameters of training
batchsize = 1
lr = 2e-4
trigger = (200000, 'iteration')
evaluate_interval = (1, 'epoch')
snapshot_interval = (1000, 'iteration')
report_interval = (1, 'iteration')

# parameters of dataset
# root = 'VCTK-Corpus'
# dataset = 'VCTK'
root = '../CMU_ARCTIC'
dataset = 'ARCTIC'
data_format = 'wav'
sr = 16000
mu = 256
top_db = 20
length = 16000

# parameters of VQ
d = 512
k = 512

# parameters of Decoder(WaveNet)
n_loop = 2
n_layer = 2
n_filter = 2
n_channel1 = 256
n_channel2 = 512
n_channel3 = 256
# quantize = 'mulaw'
quantize = 'mixture'

# parameters of losses
beta = 0.25
