# Object Segmentation Without Labels with Large-Scale Generative Models

This repository is the official implementation of ICML 2021 anonymous submission _Object Segmentation Without Labels with Large-Scale Generative Models_.
![](images/gen_scheme.jpg)\
_Schematic representation of our approach._

Core steps:
1. Find the BigBiGAN latent direction responsible for background darkening and foreground lightening;
2. Sample real images embeddings and generate segmentation masks with their shifts;
3. Train U-net with that synthetic data.

![](images/bigbigan_dog2_bg.gif)\
_example of variation along the latent direction_

## Requirements

python 3.6 or later
>torch>=1.4\
torchvision>=0.5\
tensorboardX\
scikit-image\
scipy\
h5py

> at least 2GPUs x 12Gb for batch size 95 (though smaller batch gives competitive result)

For results visualization:
>jupyter

Optional for BigBiGAN tf to torch conversion and BigBiGAN embeddings evaluation:
> tensorflow_gpu==1.15.2\
tensorflow_hub\
parse

if troubles – check the authors packages versions in ```requirements.txt```


## Embeddings and Pre-trained Models

Please download precomputed embeddings, generator weights and pretrained models from:
https://www.dropbox.com/s/wzvoigom1tscnpt/gan_saliency_data.tar?dl=0
and unpack to the repository root.

## Training

To train the U-net segmentation model with Imagenet embeddings, run this command:

```train
python train_segmentation.py \
    --out "results" \
    --gan_weights "BigGAN/weights/BigBiGAN_x1.pth" \
    --z "embeddings/BigBiGAN_ImageNet_z.npy"
    --bg_direction "BigGAN/weights/bg_direction.pth" \
    --val_images_dir __path_to_images_dir__ \
    --val_masks_dir __path_to_masks_dir__ \
```

## Custom Data

Once you want to use your own images BigBiGAN-embeddings, please run ```bigbigan_embeddings.ipynb``` notebook and provide your own data path.


## Evaluation

To check the synthetic data / segmentation samples / model metrics, please run
```inspection.ipynb``` notebook.

Here are some samples of the model performing on DUTS dataset:
![](images/duts_samples.jpg)

## BigBiGAN weights conversion
Once the original BigBiGAN weigths are distributed within tfhub, we also provide the conversion code. See ```bigbigan2biggan.ipynb```. This code is based on this script: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/TFHub/converter.py . Note that BigBiGAN and BigGAN almost share the architecture and we use the author's officially unofficial PyTorch BigGAN implementation.

## Licenses

- Unsupervised Discovery of Latent Directions: https://github.com/anvoynov/GANLatentDiscovery
- BigBiGAN weights: https://tfhub.dev/deepmind/bigbigan-resnet50/1
- BigGAN pytorch: https://github.com/ajbrock/BigGAN-PyTorch
- U-Net model code is based on: https://github.com/milesial/Pytorch-UNet
- prefetch_generator: https://github.com/justheuristic/prefetch_generator
