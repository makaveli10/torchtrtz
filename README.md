# pytorchz

This project is mainly used t generate tensorrt weights for [makaveli10/cpptensorrtz](https://github.com/makaveli10/cpptensorrtz)


## Getting Started
1. Tested with python==3.7.9
2. Install torch, torchvision
```
 $ pip install torch==1.6.0
 $ pip install torchvision==0.7.0
```
3. Install [torchsummary](https://github.com/sksq96/pytorch-summary)
```
 $ pip install torchsummary
```

## How to Run
All the models are from torchvision.
model.py will download and save the torch weights. Then gen_wts.py will write the
pytorch model in a "vgg16".wts file as required by TensorRT.

Example VGG:
```
 $ cd vgg
 $ python models.py
 $ python gen_trtwts.py
```