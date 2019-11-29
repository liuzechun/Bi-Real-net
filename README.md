# Bi-Real-net
This is the implementation of our paper "[Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://eccv2018.org/openaccess/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf)" published in ECCV 2018 and "[Bi-real net: Binarizing deep network towards real-network performance](https://arxiv.org/pdf/1811.01335.pdf)" published in IJCV. 

We proposed to use a identity mapping to propagate the real-valued information before binarization. The proposed 1-layer-per-block structure with the shortcut bypassing every binary convolutional layers significantly outperforms the original 2-layer-per-block structure in ResNet when weights and activations are binarized. The detailed motivation and discussion can be found in our IJCV paper. Three other proposed training techniques can be found in the ECCV paper.
<img width=60% src="https://github.com/liuzechun0216/images/blob/master/birealnet_figure.png"/>


# News (updated in November 23rd 2019)
We finished the pytorch implementation of training Bi-Real Net from scratch, which is super easy to run. We retrain the same accuracy as reported in the paper.
Clone and have a try with our new pytorch implementation! 

# Citation

If you use the code in your research, please cite:

    @inproceedings{liu2018bi,
      title={Bi-real net: Enhancing the performance of 1-bit cnns with improved representational capability and advanced training algorithm},
      author={Liu, Zechun and Wu, Baoyuan and Luo, Wenhan and Yang, Xin and Liu, Wei and Cheng, Kwang-Ting},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      pages={722--737},
      year={2018}
    }

and

    @article{liu2018bi,
      title={Bi-real net: Binarizing deep network towards real-network performance},
      author={Liu, Zechun and Luo, Wenhan and Wu, Baoyuan and Yang, Xin and Liu, Wei and Cheng, Kwang-Ting},
      journal={arXiv preprint arXiv:1811.01335},
      year={2018}
    }

# Pytorch Implementation 

To make Bi-Real Net easier to implement. We recently discovered that we can train it from scratch with Adam solver. The start learning rate is 0.001 and linearly decay to 0 after 256 epoches. the batchsize is set to 512. If you want to decrease or increase the batchsize, remember to multiply the learning rate with the same ratio. This implementation is different from that reported in the paper. The difference are mainly three folds:


|               |  Caffe implementation in our original paper   | Pytorch implementation   |   
| ------------- | ----- | ------------- | 
|   Training Technique    | Step-by-step finetune |     Train from scratch     |  
|   Solver     | SGD with momentum |     Adam     |  
|   Data Augmentation   | Random crop 224 from 256 | Random rescale with rescale ratio \[0.08-1\] then random crop 224 from 256 |  

Requirements:
    * python3, pytorch 1.3.0, torchvision 0.4.1


# Caffe Implementation

This model was trained on ImageNet dataset with 1000 classes and 1.2 million training images and 50k validation images. For each image in the ImageNet dataset, the smaller dimension of the image is rescaled to 256 while keeping the aspect ratio intact. For
training, a random crop of size 224 × 224 is selected. Note that, in contrast to XNOR-Net and the full-precision ResNet, we do not use the operation of random resize, which might improve the performance further. For inference, we employ the 224 × 224 center crop from images.

Pre-training: We prepare the real-valued network for initializing binary network in three steps: 1) Train the network with ReLU nonlinearity function from scratch, following the hyper-parameter settings in ResNet. 2) Replace ReLU with leaky-clip with the range of (-1,1) and the negative slope of 0.1 and finetune the network. 3) Finetune the network with clip(-1,x,1) nonlinearity instead of leaky-clip.

Training: We train two instances of the Bi-Real net, including an 18-layer Bi-Real net and a 34-layer Bi-Real net. The training of them consists of two steps: training the 1-bit convolution layer and retraining the BatchNorm. In the first step, the weights in the 1-bit convolution layer are binarized to the sign of real-valued weights multiplying the absolute mean of each kernel. We use the SGD solver with the momentum of 0.9 and set the weight-decay to 0, which means we no longer encourage the weights to be close to 0. For the 18-layer Bi-Real net, we run the training algorithm for 20 epochs with a batch size of 128. The learning rate starts from 0.01 and is decayed twice by multiplying 0.1 at the 10th and the 15th epoch. For the 34-layer Bi-Real net, the training process includes 40 epochs and the batch size is set to 1024. The learning rate starts from 0.08 and  is  multiplied  by  0.1  at  the  20th and  the  30th epoch,  respectively.  In  the second  step,  we  constraint  the  weights  to  -1  and  1,  and  set  the  learning  rate in all convolution layers to 0 and retrain the BatchNorm layer for 1 epoch to absorb the scaling factor.

Inference: we use the trained model with binary weights and binary activations in the 1-bit convolution layers for inference.

Using the code: this is a caffe implementation. We added the binary convolution layer and leaky-clip layer. The binary convolution layer is modified from https://github.com/loswensiana/BWN-XNOR-caffe, in which we modified the gradient computation method. To use the code, please put the ex_layers folder under the src and include folder respectively. Also you need to replace the original thereshold layer with our threshold layer because we modified its backward computation.

# Accuracy

|               |       | Bi-Real net   |   XNOR-Net    |
| ------------- | ----- | ------------- | ------------- |
|   18-layer    | Top-1 |     56.4%     |     51.2%     |
|               | Top-5 |     79.5%     |     73.4%     |
|   34-layer    | Top-1 |     62.2%     |               |
|               | Top-5 |     83.9%     |               |
|   50-layer    | Top-1 |     62.6%     |               |
|               | Top-5 |     83.9%     |               |


