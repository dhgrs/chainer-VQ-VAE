# parameters of training
batchsize = 8
lr = 2e-4
trigger = (100000, 'iteration')
report_interval = (5000, 'iteration')

# parameters of dataset
root = 'VCTK-Corpus'
data_format = 'wav'
sr = 16000
mu = 256
length = 16000

# parameters of VQ
d = 256
k = 384

# parameters of Decoder(WaveNet)
n_loop = 3
n_layer = 10
n_filter = 2
n_channel1 = 64
n_channel2 = 32
n_channel3 = 256

# parameters of losses
beta = 0.25
