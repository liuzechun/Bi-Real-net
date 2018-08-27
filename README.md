# Bi-Real-net
This is the caffe implementation of our paper "Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm" published in ECCV 2018. 

# Training Procedure
This model was trained on ImageNet dataset with 1000 classes and 1.2 million training images and 50k validation images. For each image in the ImageNet dataset, the smaller dimension of the image is rescaled to 256 while keeping the aspect ratio intact. For
training, a random crop of size 224 × 224 is selected. Note that, in contrast to XNOR-Net and the full-precision ResNet, we do not use the operation of random resize, which might improve the performance further. For inference, we employ the 224 × 224 center crop from images.

Pre-training: We prepare the real-valued network for initializing binary network in three steps: 1) Train the network with ReLU nonlinearity function from scratch, following the hyper-parameter settings in ResNet. 2) Replace ReLU with leaky-clip with the range of (-1,1) and the negative slope of 0.1 and finetune the network. 3) Finetune the network with clip(-1,x,1) nonlinearity instead of leaky-clip.

Training: We train two instances of the Bi-Real net, including an 18-layer Bi-Real net and a 34-layer Bi-Real net. The training of them consists of two steps: training the 1-bit convolution layer and retraining the BatchNorm. In the first step, the weights in the 1-bit convolution layer are binarized to the sign of real-valued weights multiplying the absolute mean of each kernel. We use the SGD solver with the momentum of 0.9 and set the weight-decay to 0, which means we no longer encourage the weights to be close to 0. For the 18-layer Bi-Real net, we run the training algorithm for 20 epochs with a batch size of 128. The learning rate starts from 0.01 and is decayed twice by multiplying 0.1 at the 10^{th} and the 15^{th} epoch. For the 34-layer Bi-Real net, the training process includes 40 epochs and the batch size is set to 1024. The learning rate starts from 0.08 and  is  multiplied  by  0.1  at  the  20^{th} and  the  30^{th} epoch,  respectively.  In  the second  step,  we  constraint  the  weights  to  -1  and  1,  and  set  the  learning  rate in all convolution layers to 0 and retrain the BatchNorm layer for 1 epoch to absorb the scaling factor.

Inference: we use the trained model with binary weights and binary activations in the 1-bit convolution layers for inference.

# Accuracy

|               |       | Bi-Real net   |   XNOR-Net    |
| ------------- | ----- | ------------- | ------------- |
|   18-layer    | Top-1 |     56.4%     |     51.2%     |
|               | Top-5 |     79.5%     |     73.4%     |
|   34-layer    | Top-1 |     62.2%     |     79.5%     |
|               | Top-5 |     83.9%     |               |

# Using the code
This is a caffe implementation. We added the binary convolution layer and leaky-clip layer. Put the ex_layers folder under the src and include folder respectively. We also modified the backward gradient computation for the threshold layer.
