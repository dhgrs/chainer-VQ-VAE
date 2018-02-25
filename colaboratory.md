# Train/Generate on Google Colaboratory
Chainer is a minor DL library so many reseachers can't run my code. So I prepared some notebooks. You can reproduce my results with only a browser and no need to install chainer or other packages.

## Step1 Download
You need two notebooks.

- [train.ipynb](train.ipynb) (for training)
- [generate.ipynb](generate.ipynb) (for generating)

Download each file or clone this repository.

## Spep2 Upload
Upload these notebooks on Google Drive. The dataset(CMU ARCTIC) will be downloaded into the instance of Colaboratory so it doesn't need your strage of Google Drive for dataset. But traind model's parameters, losses and etc. are saved into your Google Drive so check your starage.

## Step3 Train
Now you can train on Colaboratory. Open train.ipynb via Colaboratory and run cells. You need authentications to mount Google Drive. Please follow the indications.

You can resume training from a snapshot. Please check comments in the notebook.

## Step4 Generate
Open generate.ipynb via Colaboratory and run cells. You can use trained models in your Google Drive.
