
# SAPipe: Staleness-Aware Pipeline for Data-Parallel DNN Training

This repository is the implementation of the core part of [SAPipe: Staleness-Aware Pipeline for Data-Parallel DNN Training]. It is based on BytePS: https://github.com/bytedance/byteps.
The implementationi is partially migrated from an internal repository, and will be further updated to solve unexpected bugs. 

## Requirements

To install requirements:

```setup
cd sapipe
pip3 install torch==1.7.0 torchvision==0.8.0
wget https://raw.githubusercontent.com/mli/deps/master/build/zeromq-4.1.4.tar.gz
sudo BYTEPS_NVCC_PATH=/path/to/nvcc BYTEPS_WITH_UCX=0 python3 setup.py install
```

## Training
The training script of CIFAR-10 is in "example/pytorch/pytorch-cifar"; 

To train pipesgd, run this command:

```train
bash run_local.sh --staleness 1
```

To train the model(s) in the paper, run this command:

```train
bash run_local.sh --staleness 1  --pipesgd-weight-prediction local
```


## Results

Our model achieves the following performance on :

### [Image Classification on CIFAR-10]

| Method             | ResNet50        | VGG16            |
| ------------------ |---------------- | ---------------- |
| BytePS             |     93.2%         |  92.5%           |
| PipeSGD            |     89.3%         |  90.6%           | 
| SAPipe             |     93.2%         |  92.6%           |

