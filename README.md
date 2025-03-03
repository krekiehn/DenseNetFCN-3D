# DenseNet3D and DenseNet3D-FCN models for Keras.

Adapted DenseNet and DensetNet-FCN to work with 3D input for volume classification and segmentation.

DenseNet is a network architecture where each layer is directly connected
to every other layer in a feed-forward fashion (within each dense block).
For each layer, the feature maps of all preceding layers are treated as
separate inputs whereas its own feature maps are passed on as inputs to
all subsequent layers. This connectivity pattern yields state-of-the-art
accuracies on CIFAR10/100 (with or without data augmentation) and SVHN.
On the large scale ILSVRC 2012 (ImageNet) dataset, DenseNet achieves a
similar accuracy as ResNet, but using less than half the amount of
parameters and roughly half the number of FLOPs.

DenseNets can be extended to image segmentation tasks as described in the
paper "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for
Semantic Segmentation". Here, the dense blocks are arranged and concatenated
with long skip connections for state of the art performance on the CamVid dataset.

## Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic
   Segmentation](https://arxiv.org/pdf/1611.09326.pdf)

This implementation is based on the following reference code:
 - https://github.com/gpleiss/efficient_densenet_pytorch
 - https://github.com/liuzhuang13/DenseNet
 - https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/densenet.py
 
 ## Tutorial
 - https://towardsdatascience.com/implementing-a-fully-convolutional-network-fcn-in-tensorflow-2-3c46fb61de3b
 - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9167192
 
 - https://www.nature.com/articles/s41598-020-70479-z/tables/2

 ## DOCKER COMMANDS
docker run -u $(id -u):$(id -g) -it --gpus=3 --ipc=host -v $(pwd):/code -v /home/sukin699/Verse/Verse_full:/data tensorflow/tensorflow  
