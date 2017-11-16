# chainer-VQ-VAE
A Chainer implementation of VQ-VAE( https://arxiv.org/abs/1711.00937 ).

# Requirements
- python3
- chainer v3
- librosa

# Usage
## set parameters
Edit `opt.py` before training. Maybe you have to change `root` to the directory that you download VCTK-Corpus. You can download VCTK-Corpus from [here](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html).

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

## generating
```
python generate.py <trained model> <input audio>
```

# TODO
- [ ] upload generated sample(now training and it's progressing well)
- [ ] speaker conditional model
