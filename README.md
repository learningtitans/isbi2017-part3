# ISIC 2017 Challenge Models by "RECOD Titans"

This repository is a branch of [Tensorflow/models/slim](https://github.com/tensorflow/models/tree/master/research/slim) containing the
models implemented by [RECOD "Titans"](https://recodbr.wordpress.com/) for the IEEE ISBI 2017 Challenge presented by ISIC ([ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a) challenge / Part 3:
Lesion Classification).

RECOD "Titans" got the best ROC AUC for melanoma classification (87.4%), 3rd best ROC AUC for seborrheic keratosis classification (94.3%), and 3rd best combined/mean ROC AUC (90.8%).

There's a separated repository for the [models used in Part 1: Lesion Segmentation](https://github.com/learningtitans/isbi2017-part1), and a [technical report](https://arxiv.org/abs/1703.04819) detailing our participation on both tasks.

## Foreword

**Please note: this is a beta public release**. Please, help us to improve this code, by [submitting an issue](https://github.com/learningtitans/isbi2017-part3/issues) if you find any problems.

Despite the best effort of authors, reproducing results of todays' Machine Learning is challenging, due to the complexity of the machinery, involving millions of lines of code distributed among thousands of packages — and the management of hundreds of random factors.

We are committed to alleviate that problem. We are a very small team, and unfortunately, cannot provide help with technical issues (e.g., procuring data, installing hardware or software, etc.), but we'll do our best to share the technical and scientific details needed to reproduce the results. Please, see our contacts at the end of this documents.

Most of the code is a direct copy of the models posted in Tensorflow/slim, adjusted to fit the challenge (dataset, data preparation, results formatting, etc.). We created the code needed for the SVM decision layers, and the final meta-learning SVM stacking.

N.B.: Our code is now *a lot* behind current Tensorflow/slim. If you need to contrast our code with a reference version, the [March 1st 2017 commit](https://github.com/tensorflow/models/commit/6a9c0da96295f45909472cc70674b5b7d5c6fc2d) is a good place to start. In order to run our code you don't have to download Tensorflow/slim, you just need Tensorflow-GPU v.012 (it won't work with newer versions of Tensorflow after r1.0).

**If you use this code in an academic context, please cite us.** The main reference is the "RECOD Titans at ISIC Challenge 2017" report. If the transfer learning aspects of this work are important to your context, you might find appropriate to cite the ISBI 2017 paper "Knowledge transfer for melanoma screening with deep learning" as well. The report and the paper are linked at the end of this file.

## Requirements

*Hardware:* You'll need a CUDA/cuDNN compatible GPU card with enough RAM. We tested our models on NVIDIA GeForce Titan X, Titan X (Pascal), and Tesla K40c cards, all with 12 GiB of RAM.

*Software:* All our tests used Linux. We ran most experiments on Ubuntu 14.04.5 LTS, and some on Debian 5.4.1-4. You'll needed Python 2.7, with packages tensorflow-gpu (v0.12), numpy, scipy, and sklearn. You can install those packages with pip. You'll also need curl, git, ImageMagick, and bc (the command-line basic calculator).

*Docker installation:* The easiest way to install the needed software is to create a [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) container from image tensorflow/tensorflow:0.12.1-gpu, and then add the remaining packages:

```bash
nvidia-docker pull tensorflow/tensorflow:0.12.1-gpu

mkdir ~/isbi2017-part3

nvidia-docker run -ti -e OUTSIDE_USER=$USER  -e OUTSIDE_UID=$UID -e OUTSIDE_GROUP=`/usr/bin/id -ng $USER` -e OUTSIDE_GID=`/usr/bin/id -g $USER` -v $HOME/isbi2017-part3:/isbi2017-part3 --name isbichallenge2017 tensorflow/tensorflow:0.12.1-gpu /bin/bash

# Inside container:
apt-get update
apt-get install git -y
apt-get install imagemagick -y
apt-get install bc -y

groupadd --gid "$OUTSIDE_GID" "$OUTSIDE_GROUP"
useradd --create-home --uid "$OUTSIDE_UID" --gid "$OUTSIDE_GID" "$OUTSIDE_USER"

su -l $OUTSIDE_USER
ln -s /isbi2017-part3 ~/isbi2017-part3
```

The procedure above creates a user inside the container equivalent to your external user, and maps the external directory ~/isbi2017-part3 into /isbi2017-part3 inside the container. That is highly recommended because Docker filesystem isn't fit for extensive data manipulation.

## Cloning this repository

We're assuming that you'll use the commands below to clone this repository. If you use a different path than ~/isbi2017-part3, adapt the instructions that follow as needed.

```bash
cd ~/
git clone https://github.com/learningtitans/isbi2017-part3
```

## Obtaining and preparing the data

External data was allowed by the challenge. We collected data from several sources, listed below. Those sources are publicly obtainable — more or less easily — some requiring a license agreement, some requiring payment, some requiring both.

If you're going to use our pre-trained models (see below) and test on the official challenge validation and test splits, you just have to procure the official challenge datasets and prepare the test sets (see below). If you want to train the models "from scratch" and reproduce our steps exactly, you'll have to procure all datasets, and prepare the training set as well.

### Official challenge datasets

The official [ISIC 2017 Challenge](https://challenge.kitware.com/#phase/5840f53ccad3a51cc66c8dab) challenge dataset has 2,000 dermoscopic images (374 melanomas, 254 seborrheic keratoses, and 1,372 benign nevi). It's freely available, after signing up at the challenge website. **You have to download both training and test (and validation, if desired) sets**, and unzip them to the ~/isbi2017-part3/data/challenge directory (flat in the same directory, or in subdirectories, it does not matter).

*The procedure:* (1) Download/unzip the challenge files into ~/isbi2017-part3/data/challenge (including ground truth data). (2) Delete or move the superpixel files. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/challenge
cd ~/isbi2017-part3/data/challenge

# Download and unzip all files here

mkdir -p ~/isbi2017-part3/extras/challenge/superpixels
find . -name '*.png' -exec mv -v "{}" ~/isbi2017-part3/extras/challenge/superpixels \;
```

### Additional ISIC Archive Images

We used additional images from the [ISIC Archive](http://isdis.net/isic-project/) an international consortium to improve melanoma diagnosis, containing  over 13,000 dermoscopic images. They're freely available. We used a relatively small subset of the Archive.

*The procedure:* Download the images to ~/isbi2017-part3/data/isic. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/isic/images
cd ~/isbi2017-part3/data/isic/images
cat ~/isbi2017-part3/data/isic-ids.txt | while read imgid; do curl -o $imgid.jpg https://isic-archive.com:443/api/v1/image/$imgid/download?contentDisposition=attachment; sleep 1; done
```

### Interactive Atlas of Dermoscopy

The [Interactive Atlas of Dermoscopy](http://www.dermoscopy.org/) has 1,000+ clinical cases (270 melanomas, 49 seborrheic keratoses), each with at least two images: dermoscopic, and close-up clinical. It's available for anyone to buy for ~250€.

*The procedure:* (1) Insert/Mount the Atlas CD-ROM. (2) Copy all image files to ~/isbi2017-part3/data/atlas. (3) Rename all files to lowercase. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/atlas
cd ~/isbi2017-part3/data/atlas
# Adapt the path /media/cdrom below to the CD mount point
find /media/cdrom/Images -name '*.jpg' -exec sh -c 'cp -v "{}" `basename "{}" | tr "[A-Z]" "[a-z]"`' \;
```

### Dermofit Image Library

The [Dermofit Image Library](https://licensing.eri.ed.ac.uk/i/software/dermofit-image-library.html) has 1,300 images (76 melanomas, 257 seborrheic keratoses). It's available after signing a license agreement, for a fee of ~50€.

*The procedure:* (1) Download/unzip all the dataset files (*.zip) to ~/isbi2017-part3/data/dermofit. (2) Delete or move the mask files. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/dermofit
cd ~/isbi2017-part3/data/dermofit

# Download and unzip all files here

mkdir -p ~/isbi2017-part3/extras/dermofit/masks
find . -name '*mask.png' -exec mv -v "{}" ~/isbi2017-part3/extras/dermofit/masks \;
```

### The IRMA Skin Lesion Dataset

The [IRMA Skin Lesion Dataset](http://ganymed.imib.rwth-aachen.de/irma/datasets/)
has 747 dermoscopic images (187 melanomas). This dataset is unlisted, but available under special request, and the signing of a license agreement.

*The procedure:* Download/unzip all the images to ~/isbi2017-part3/data/irma. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/irma
cd ~/isbi2017-part3/data/irma
# Download all images here
```

### The PH2 Dataset

The [PH2 Dataset](http://www.fc.up.pt/addi/ph2%20database.html) has 200 dermoscopic images (40 melanomas). It's freely available after signing a short online registration form.

*The procedure:* (1) Download/unzip all the images to ~/isbi2017-part3/data/ph2. (2) Delete or move the maskfiles. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/ph2
cd ~/isbi2017-part3/data/ph2

# Download all images here

mkdir -p ~/isbi2017-part3/extras/ph2/masks
find . -name '*_mask.bmp' -exec mv -v "{}" ~/isbi2017-part3/extras/ph2/masks \;
```

### Integrating the dataset

*The procedure:* (1) Copy all images to a single folder, while resizing them to 299×299, and converting them to JPEG. (2) Repeat on another folder for a size of 224×224. One way to accomplish it:

```bash
mkdir -p ~/isbi2017-part3/data/images299
cd ~/isbi2017-part3/data/images299
find ~/isbi2017-part3/data -name '*.jpg' -exec sh -c 'echo "{}"; convert "{}" -resize 299x299\! `basename "{}"`' \;
find ~/isbi2017-part3/data -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 299x299\! `basename "{}" .png`.jpg' \;
find ~/isbi2017-part3/data -name '*.bmp' -exec sh -c 'echo "{}"; convert "{}" -resize 299x299\! `basename "{}" .bmp`.jpg' \;

mkdir -p ~/isbi2017-part3/data/images224
cd ~/isbi2017-part3/data/images224
find ~/isbi2017-part3/data -name '*.jpg' -exec sh -c 'echo "{}"; convert "{}" -resize 224x224\! `basename "{}"`' \;
find ~/isbi2017-part3/data -name '*.png' -exec sh -c 'echo "{}"; convert "{}" -resize 224x224\! `basename "{}" .png`.jpg' \;
find ~/isbi2017-part3/data -name '*.bmp' -exec sh -c 'echo "{}"; convert "{}" -resize 224x224\! `basename "{}" .bmp`.jpg' \;
```

### Converting training images and metadata to Tensorflow TF-Record format

The procedure below creates the actual training sets. The training set we called "deploy" in the technical report contains all images listed in the first column of ~/isbi2017-part3/data/deploy2017.txt. The training set we called "semi" contain those images, minus those listed in ~/isbi2017-part3/data/diff-semi.txt.

Each training set will actually be separated into three splits: train, validation, and a vestigial test split with a handful of images (due to our reuse of generic code that assumes three splits). The train split was used to find the weights in the deep learning models, and to train the SVM layers in the models which use it. The validation split was used to compute what we called in the report "internal validation AUC", to train the stacked SVM meta-model, and — in a few cases — to establish an early-stopping procedure for the deep-learning training. We didn't use the vestigial test split.

You can inspect which images fall in which splits by listing the *.log files in the *.tfr folders created below.

The procedure:

```bash
mkdir -p ~/isbi2017-part3/data/deploy.299.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TRAIN ~/isbi2017-part3/data/deploy2017.txt ~/isbi2017-part3/data/images299 ~/isbi2017-part3/data/deploy.299.tfr ~/isbi2017-part3/data/no-blacklist.txt

mkdir -p ~/isbi2017-part3/data/semi.299.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TRAIN ~/isbi2017-part3/data/deploy2017.txt ~/isbi2017-part3/data/images299 ~/isbi2017-part3/data/semi.299.tfr ~/isbi2017-part3/data/diff-semi.txt

mkdir -p ~/isbi2017-part3/data/deploy.224.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TRAIN ~/isbi2017-part3/data/deploy2017.txt ~/isbi2017-part3/data/images224 ~/isbi2017-part3/data/deploy.224.tfr ~/isbi2017-part3/data/no-blacklist.txt

mkdir -p ~/isbi2017-part3/data/semi.224.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TRAIN ~/isbi2017-part3/data/deploy2017.txt ~/isbi2017-part3/data/images224 ~/isbi2017-part3/data/semi.224.tfr ~/isbi2017-part3/data/diff-semi.txt
```

### Preparing the official test dataset

The procedure below creates the official test sets to tf-record:

```bash
mkdir -p ~/isbi2017-part3/data/test.299.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TEST ~/isbi2017-part3/data/isbi2017_official_test_v2.txt ~/isbi2017-part3/data/images299 ~/isbi2017-part3/data/test.299.tfr ~/isbi2017-part3/data/no-blacklist.txt

mkdir -p ~/isbi2017-part3/data/test.224.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TEST ~/isbi2017-part3/data/isbi2017_official_test_v2.txt ~/isbi2017-part3/data/images224 ~/isbi2017-part3/data/test.224.tfr ~/isbi2017-part3/data/no-blacklist.txt
```

### (Optional) Preparing the official validation dataset

The procedure below converts the official validation sets to tf-record:

```bash
mkdir -p ~/isbi2017-part3/data/val.299.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TEST ~/isbi2017-part3/data/isbi2017_official_validation.txt ~/isbi2017-part3/data/images299 ~/isbi2017-part3/data/val.299.tfr ~/isbi2017-part3/data/no-blacklist.txt

mkdir -p ~/isbi2017-part3/data/val.224.tfr
python ~/isbi2017-part3/datasets/convert_skin_lesions.py TEST ~/isbi2017-part3/data/isbi2017_official_validation.txt ~/isbi2017-part3/data/images224 ~/isbi2017-part3/data/val.224.tfr ~/isbi2017-part3/data/no-blacklist.txt
```

## Pre-trained model

We released a pre-trained, ready-to-use model. That model is exactly the one used in the Challenge, so modulo bugs, package incompatibilites and random fluctuations, you should get the same AUCs as we did.

The pre-trained model consists of 3 base deep-learning models, 3 base-model SVM layers, and one final stacked SVM layer. That's _a lot_ of parameters! The files are too big for github, and are shared on [figshare](https://figshare.com/) with [DOI: 10.6084/m9.figshare.4993931](http://dx.doi.org/10.6084/m9.figshare.4993931). Direct links are provided below:

| File | Size | MD5 |
| -- | :--: | :--: |
| [Deep Learning Model RC.25](https://ndownloader.figshare.com/files/8411216) | 465M |eaf7bb10806783d54c6b72c90b42486e |
| [Deep Learning Model RC.28](https://ndownloader.figshare.com/files/8411219) | 465M |700d7ef0ee53e6729e8a20bdd1acf8d8 |
| [Deep Learning Model RC.30](https://ndownloader.figshare.com/files/8411222) | 453M |360167526d3a52fc52f7f4dced5035f1 |
| [All SVM Models](https://ndownloader.figshare.com/files/8411225) | 71M | ce79acca7cf7dcdeabec32ed58e4feca |

Download and unzip all files into ~/isbi2017-part3/running...

```bash
mkdir -p ~/isbi2017-part3/running
# Download and unzip all files here
mkdir ~/isbi2017-part3/running/checkpoints.rc30/best # Fix path issue with pre-trained rc.30 model
ln -s ~/isbi2017-part3/running/checkpoints.rc30/model.ckpt-22907.* ~/isbi2017-part3/running/checkpoints.rc30/best
```

...then proceed to [predicting with the model](#predicting-with-the-model).

## Training the model "from scratch"

Strictly speaking, the training will not be purely from scratch, since we will transfer knowledge from models pre-trained on ImageNet. We do not recommend — except for scientific curiosity — training strictly from scratch, since training for ImageNet is a slow and complex endeavor in itself.

We need the ImageNet weights of two models: Resnet-101 and Inception-v4, available [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) (or check the exact addresses at the curl commands below). Download and unzip them to ~/isbi2017-part3/running:

```bash
mkdir -p ~/isbi2017-part3/running
cd ~/isbi2017-part3/running
curl http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz | tar xvz
curl http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz | tar xvz
```

### Deep Learning Component Model rc25

Inception v4 trained on "deploy" dataset for 40000 batches, with per-image normalization that erases the average of the pixels.

```bash
mkdir -p ~/isbi2017-part3/running/checkpoints.rc25
cd ~/isbi2017-part3/
python train_image_classifier.py \
    --train_dir=$HOME/isbi2017-part3/running/checkpoints.rc25 \
    --dataset_dir=$HOME/isbi2017-part3/data/deploy.299.tfr \
    --dataset_name=skin_lesions \
    --task_name=label \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --preprocessing_name=dermatologic \
    --checkpoint_path=$HOME/isbi2017-part3/running/inception_v4.ckpt  \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --save_interval_secs=3600 \
    --optimizer=rmsprop \
    --normalize_per_image=1 \
    --max_number_of_steps=40000 \
    --experiment_tag="Model: Inceptionv4 Train: Deploy; Normalization: mode 1, erases mean"  \
    --experiment_file=$HOME/isbi2017-part3/running/checkpoints.rc25/experiment.meta
```

### Deep Learning Component Model rc28

Inception v4 trained on "semi" dataset for 40000 batches, with per-image normalization that erases the average of the pixels.

```bash
mkdir -p ~/isbi2017-part3/running/checkpoints.rc28
cd ~/isbi2017-part3/
python train_image_classifier.py \
    --train_dir=$HOME/isbi2017-part3/running/checkpoints.rc28 \
    --dataset_dir=$HOME/isbi2017-part3/data/semi.299.tfr \
    --dataset_name=skin_lesions \
    --task_name=label \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --preprocessing_name=dermatologic \
    --checkpoint_path=$HOME/isbi2017-part3/running/inception_v4.ckpt  \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --save_interval_secs=3600 \
    --optimizer=rmsprop \
    --normalize_per_image=1 \
    --max_number_of_steps=40000 \
    --experiment_tag="Train: Semi; Normalization: mode 1, erases mean"  \
    --experiment_file=$HOME/isbi2017-part3/running/checkpoints.rc28/experiment.meta
```

### Deep Learning Component Model rc30

Resnet-101 v1 trained on "semi" dataset for 40000 batches, with per-image normalization that erases the average of the pixels.

```bash
mkdir -p ~/isbi2017-part3/running/checkpoints.rc30/best
cd ~/isbi2017-part3/
python train_image_classifier.py \
      --train_dir=$HOME/isbi2017-part3/running/checkpoints.rc30 \
      --dataset_dir=$HOME/isbi2017-part3/data/semi.224.tfr \
      --dataset_name=skin_lesions \
      --task_name=label \
      --dataset_split_name=train \
      --train_image_size=224 \
      --model_name=resnet_v1_101 \
      --preprocessing_name=vgg \
      --checkpoint_path=$HOME/isbi2017-part3/running/resnet_v1_101.ckpt  \
      --checkpoint_exclude_scopes=resnet_v1_101/logits \
      --save_interval_secs=3600 \
      --normalize_per_image=1 \
      --max_number_of_steps=40000 \
      --experiment_tag="Network: Resnet101"  \
      --experiment_file=$HOME/isbi2017-part3/running/checkpoints.rc30/experiment.meta
```

This is the only model that requires validation for early stopping. The validation loop has to run as a separate process. You have to run the command below *at the same time* as the training above running (for example, in another shell):

```bash
cd ~/isbi2017-part3/
./etc/launch_validation_loop.sh RESNET \
      $HOME/isbi2017-part3/running/checkpoints.rc30 \
      $HOME/isbi2017-part3/data/semi.224.tfr
```

### Training the SVM layer of the component models

We start by extracting the needed features from the training set:

```bash
mkdir -p ~/isbi2017-part3/running/svm.features
cd ~/isbi2017-part3/
python predict_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=$HOME/isbi2017-part3/running/checkpoints.rc25/model.ckpt-40000 \
    --dataset_dir=$HOME/isbi2017-part3/data/deploy.299.tfr \
    --dataset_name=skin_lesions \
    --task_name=label \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --preprocessing_name=dermatologic \
    --id_field_name=id \
    --eval_replicas=50 \
    --pool_features=none \
    --pool_scores=none \
    --extract_features \
    --add_scores_to_features=logits \
    --output_file=$HOME/isbi2017-part3/running/svm.features/train.50.rc25.feats \
    --output_format=pickle \
    --normalize_per_image=1

python predict_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=$HOME/isbi2017-part3/running/checkpoints.rc28/model.ckpt-40000 \
    --dataset_dir=$HOME/isbi2017-part3/data/semi.299.tfr \
    --dataset_name=skin_lesions \
    --task_name=label \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --preprocessing_name=dermatologic \
    --id_field_name=id \
    --eval_replicas=50 \
    --pool_features=none \
    --pool_scores=none \
    --extract_features \
    --add_scores_to_features=logits \
    --output_file=$HOME/isbi2017-part3/running/svm.features/train.50.rc28.feats \
    --output_format=pickle \
    --normalize_per_image=1

python etc/aggregate_pickle.py \
    $HOME/isbi2017-part3/running/svm.features/train.50.rc28.feats \
    $HOME/isbi2017-part3/running/svm.features/train.50avg.rc28.feats
```

Then, we train the SVM models. The train_svm_layer.py script below uses multi-threading to accelerate the training, but that has pitfalls due to [joblib](https://pythonhosted.org/joblib/) dealing poorly with temporary files over NFS. If you experience issues, set the environment variable JOBLIB_TEMP_FOLDER to a local directory, or change ```--jobs 4``` to ```--jobs 1``` (the training will be significantly slower, but still tolerable).

```bash
mkdir -p ~/isbi2017-part3/running/svm.models
cd ~/isbi2017-part3/

python train_svm_layer.py --input_training ~/isbi2017-part3/running/svm.features/train.50.rc25.feats --output_model ~/isbi2017-part3/running/svm.models/rc25.50.svm --jobs 4 --svm_method LINEAR_PRIMAL

python train_svm_layer.py --input_training ~/isbi2017-part3/running/svm.features/train.50.rc28.feats --output_model ~/isbi2017-part3/running/svm.models/rc28.50.svm --jobs 4 --svm_method LINEAR_PRIMAL

python train_svm_layer.py --input_training ~/isbi2017-part3/running/svm.features/train.50avg.rc28.feats --output_model ~/isbi2017-part3/running/svm.models/rc28.50avg.svm --jobs 4 --svm_method LINEAR_PRIMAL
```

### Training the final SVM meta-model

We start by making the predictions in the internal validation dataset, from which the meta-model will be trained:

```bash
mkdir -p ~/isbi2017-part3/running/meta.training
cd ~/isbi2017-part3

./etc/predict_all_component_models_isbi.sh ~/isbi2017-part3/data/deploy.299.tfr ~/isbi2017-part3/data/deploy.224.tfr validation ~/isbi2017-part3/running/meta.training
```

Each component model is sampled thrice. The procedure below creates 100 replicas from combinations of those samples:

```bash
python etc/assemble_meta_features.py ALL_LOGITS ~/isbi2017-part3/running/meta.training ~/isbi2017-part3/running/svm.features/validation.metall.feats ~/isbi2017-part3/data/deploy2017.txt
```

Finally, the stacked SVM model is learned (the warnings above about joblib apply here as well):

```bash
python train_svm_layer.py --input_training ~/isbi2017-part3/running/svm.features/validation.metall.feats --output_model ~/isbi2017-part3/running/svm.models/metall.svm --jobs 4 --svm_method LINEAR_PRIMAL --max_iter_hyper 30 --preprocess NONE
```

## Predicting with the model

The instructions below show how to make predictions with the model, by assembling the official challenge submissions.

Start by getting the predictions and features from the componente models:

```bash
mkdir -p ~/isbi2017-part3/running/isbitest.features
cd ~/isbi2017-part3

./etc/predict_all_component_models_isbi.sh ~/isbi2017-part3/data/test.299.tfr ~/isbi2017-part3/data/test.224.tfr test ~/isbi2017-part3/running/isbitest.features
```

Each component model is sampled thrice. The procedure below creates 100 replicas from combinations of those samples:

```bash
python etc/assemble_meta_features.py ALL_LOGITS ~/isbi2017-part3/running/isbitest.features ~/isbi2017-part3/running/isbitest.features/isbitest.metall.features
```

Finally, get the predictions from the stacked meta-model:

```bash
mkdir -p ~/isbi2017-part3/submission/
python predict_svm_layer.py \
    --input_model ~/isbi2017-part3/running/svm.models/metall.svm  \
    --input_test ~/isbi2017-part3/running/isbitest.features/isbitest.metall.features \
    --pool_by_id xtrm \
    > ~/isbi2017-part3/submission/isbi2017-rc36xtrm.txt
```

### (Optional) Predicting for the official validation set

Change the commands above to:

```bash
mkdir -p ~/isbi2017-part3/running/isbival.features
cd ~/isbi2017-part3

./etc/predict_all_component_models_isbi.sh ~/isbi2017-part3/data/val.299.tfr ~/isbi2017-part3/data/val.224.tfr test ~/isbi2017-part3/running/isbival.features

python etc/assemble_meta_features.py ALL_LOGITS ~/isbi2017-part3/running/isbival.features ~/isbi2017-part3/running/isbival.features/isbival.metall.features

mkdir -p ~/isbi2017-part3/submission/
python predict_svm_layer.py \
    --input_model ~/isbi2017-part3/running/svm.models/metall.svm  \
    --input_test ~/isbi2017-part3/running/isbival.features/isbival.metall.features \
    --pool_by_id xtrm \
    > ~/isbi2017-part3/submission/isbi2017-val-rc36xtrm.txt
```

## Checking the procedure

There are two tests you can apply to check if you've ran the procedures correctly: contrast your submission files with ours, and check your submission files against the the challenge ground truth.

### Comparing the submission files

You cannot just diff the submission files, because the probabilities will be slightly different (due to random factors we did not control in the procedure above). However, the classification order should be the same, or almost the same between runs. Not all inversions are significant (ranks inversions between images on the same class do not affect the metrics).

You can check your files agains ours using the commands below:

```bash
cd ~/isbi2017-part3

python etc/count_inversions.py data/challenge/ISIC-2017_Test_v2_Part3_GroundTruth.csv data/isbi2017-titans-testv2-rc36xtrm.txt submission/isbi2017-rc36xtrm.txt
python etc/count_inversions.py data/challenge/ISIC-2017_Validation_Part3_GroundTruth.csv data/isbi2017-titans-val-rc36xtrm.txt submission/isbi2017-val-rc36xtrm.txt
```

In our tests we found that a few thousand significant inversions are expected.


### Comparing the performances

To compare performances, first download to ~/isbi2017-part3/data the ground truth files for the challenge [test set](https://challenge.kitware.com/#phase/584b0afccad3a51cc66c8e38) and for the challenge [validation set](https://challenge.kitware.com/#phase/584b0afacad3a51cc66c8e33). Then run the commands below:

```bash
cd ~/isbi2017-part3

python etc/compute_metrics.py data/challenge/ISIC-2017_Test_v2_Part3_GroundTruth.csv submission/isbi2017-rc36xtrm.txt
python etc/compute_metrics.py data/challenge/ISIC-2017_Validation_Part3_GroundTruth.csv submission/isbi2017-val-rc36xtrm.txt
```

For the test set you should get numbers very close to:
```
Melanoma AUC: 0.873511
Keratosis AUC: 0.942527
Average AUC: 0.908019
```

For the validation set you should get numbers very close to:
```
Melanoma AUC: 0.907778
Keratosis AUC: 0.994929
Average AUC: 0.951354
```

In our tests we found that ~1 p.p. fluctuations are expected.

## About us

The Learning Titans are a team of researchers lead by [Prof. Eduardo Valle](http://eduardovalle.com/) and hosted by the [RECOD Lab](https://recodbr.wordpress.com/), at the [University of Campinas](http://www.unicamp.br/), in Brazil.


### Our papers and reports

A Menegola, J Tavares, M Fornaciali, LT Li, S Avila, E Valle. RECOD Titans at ISIC Challenge 2017. [arXiv preprint arXiv:1703.04819](https://arxiv.org/abs/1703.04819) | [Video presentation](https://www.youtube.com/watch?v=DFrJeh6LkE4) | [PDF Presentation](http://eduardovalle.com/wordpress/wp-content/uploads/2017/05/menegola2017isbi-RECOD-ISIC-Challenge-slides.pdf)

A Menegola, M Fornaciali, R Pires, FV Bittencourt, S Avila, E Valle. Knowledge transfer for melanoma screening with deep learning. IEEE International Symposium on Biomedical Images (ISBI) 2017. [arXiv preprint arXiv:1703.07479](https://arxiv.org/abs/1703.07479) | [Video presentation](https://www.youtube.com/watch?v=upJApUVCWJY)
| [PDF Presentation](http://eduardovalle.com/wordpress/wp-content/uploads/2017/05/menegola2017isbi-TransferLearningMelanomaScreening-slides.pdf)

M Fornaciali, M Carvalho, FV Bittencourt, S Avila, E Valle. Towards automated melanoma screening: Proper computer vision & reliable results. [arXiv preprint arXiv:1604.04024](https://arxiv.org/abs/1604.04024).

M Fornaciali, S Avila, M Carvalho, E Valle. Statistical learning approach for robust melanoma screening. SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI) 2014. [DOI: 10.1109/SIBGRAPI.2014.48](https://scholar.google.com.br/scholar?cluster=3052571560066780582&hl=en&as_sdt=0,5&sciodt=0,5) | [PDF Presentation](https://sites.google.com/site/robustmelanomascreening/SIBGRAPI_Slides_MichelFornaciali.pdf?attredirects=0)

[Robust Melanoma Screening Minisite](https://sites.google.com/site/robustmelanomascreening/)


## Copyright and license

Please check files LICENSE/AUTHORS, LICENSE/CONTRIBUTORS, and LICENSE/LICENSE.

