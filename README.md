# chainer-VQ-VAE
A Chainer implementation of VQ-VAE( https://arxiv.org/abs/1711.00937 ).

# Results
Trained 165000 iterations on CMU ARCTIC. You can reproduce these results in Google Colaboratory.

Losses:

![loss1](loss1.png)
![loss2](loss2.png)
![loss3](loss3.png)

Audios:

[Input](http://nana-music.com/sounds/037eb33f/])

[Target speaker](http://nana-music.com/sounds/0383457c/)

[Reconstruct(decode with input speaker)](http://nana-music.com/sounds/037eb451/)

[Voice Conversion(decoce with target speaker)](http://nana-music.com/sounds/037eb39a/)

# Requirements
I trained and generated with

- python(3.5.2)
- chainer(4.0.0b3)
- librosa(0.5.1)

And now you can try it on Google Colaboratory. You don't need install chainer/librosa or buy GPUs. Check [this](colaboratory.md).
# Usage
## download dataset
You can download VCTK-Corpus from [here](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html). And you can download CMU-ARCTIC very easily via [my repository](https://github.com/dhgrs/download_dataset).

## set parameters
### parameters of training
- batchsize
    - Batch size.
- lr
    - Learning rate.
- trigger
    - How many times you update the model. You can set this parameter like as (`<int>`, 'iteration') or (`<int>`, 'epoch')
- evaluate_interval
    - The interval that you evaluate validation dataset. You can set this parameter like as trigger.
- snapshot_interval
    - The interval that you save snapshot. You can set this parameter like as trigger.
- report_interval
    - The interval that you write log of loss. You can set this parameter like as trigger.

### parameters of dataset
- root
    - The root directory of training dataset.
- dataset
    - The architecture of the directory of training dataset. Now this parameter supports `VCTK` and `ARCTIC`
- data_format
    - The file format of files in training dataset. You can use formats which librosa supports.
- sr
    - Sampling rate. If it's different from input file, be resampled by librosa.
- mu
    - The parameter of Mu-Law encoding.
- top_db
    - The threshold db for triming silence.
- length
    - How many samples used for training.

### parameters of VQ
- d
    - The parameter `d` in the paper.
- k
    - The parameter `k` in the paper.

### parameters of Decoder(WaveNet)
- n_loop
    - If you want to make network like dilations [1, 2, 4, 1, 2, 4] set `n_loop` as `2`.
- n_layer
    - If you want to make network like dilations [1, 2, 4, 1, 2, 4] set `n_layer` as `3`.
- n_filter
    - The filter size of each dilated convolution. Now supports only `2`.
- residual_channels
    - The number of input/output channels of residual blocks.
- dilated_channels
    - The number of output channels of causal dilated convolution layers. This is splited into tanh and sigmoid so the number of hidden units is half of this number.
- skip_channels
    - The number of channels of skip connections and last projection layer.
- embed_channels
    - The dimension of speaker embeded-vector.

### parameters of losses
- beta
    - The parameter `beta` in the paper.

## training
```
(without GPU)
python train.py

(with GPU #n)
python train.py -g n
```

If you want to use multi GPUs, you can add IDs like below.
```
python train.py -g 0 1 2
```

You can resume snapshot and restart training like below.
```
python train.py -r snapshot_iter_100000
```
Other arguments `-f` and `-p` are parameters for multiprocess in preprocessing. `-f` means the number of prefetch and `-p` means the number of processes.

## generating
```
python generate.py -i <input file> -o <output file> -m <trained model> -s <speaker>
```

If you don't set `-o`, default file name `result.wav` is used. If you don't set `-s`, the speaker is same as input file that got from filepath.

# TODO
- [x] upload generated sample
    - Current uploaded sample is old version and very poor quality. Now training newest parameters and getting good results. Please wait!
- [x] using GPU fot generating
    - Now only CPU is used for generating.
- [ ] descritized mixture of logistics
- [ ] Parallel WaveNet