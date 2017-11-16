# parameters of training
batchsize = 32
lr = 2e-4
trigger = (100000, 'iteration')
report_interval = (1000, 'iteration')

# parameters of dataset
root = 'VCTK-Corpus'
data_format = 'wav'
sr = 8000
mu = 256
length = 8192

# parameters of VQ
d = 256
k = 512

# parameters of Decoder(WaveNet)
n_loop = 3
n_layer = 10
n_filter = 2
n_channel1 = 32
n_channel2 = 16
n_channel3 = 512

# parameters of losses
beta = 0.25
