# Shadow detection of video



## Requirement
* Python 2.7
* PyTorch 0.4.0
* torchvision
* numpy
* Cython
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)

## Usage

### Training
1. Run by ```python train.py```

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently 
change it as you need.

### Testing
1. Put the trained model in ckpt/BDRAR
2. Run by ```python infer.py```

