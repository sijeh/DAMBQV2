# DAMBQ V2

Reconstructed code of CVPR2021 paper ["Distribution-aware Adaptive Multi-bit Quantization"](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Distribution-Aware_Adaptive_Multi-Bit_Quantization_CVPR_2021_paper.pdf)

## Note 
* In order to increase the extensibility and make it more elegant, we reconstructed the algorithm in this repo. The original repo could be found [here](https://github.com/sijeh/DAMBQ) 

* `model/quant_layers.py` is the module of DMBQ in the paper which will be used to replace the original `nn.Conv2d` or `nn.Linear`

* `model/quant_tune.py` is used to automate the above process and ajust the bitwidth of wights and activations in layer-wise or channel-wise (Loss-aware Bit-width Allocation, LBA)

* Only a few lines of code are needed to quantize a CNN, please refer to `main_cifar10.py` for detail and example.

## Usage
**Quantize ResNet20 with CIFAR10 dataset for example**

1. Train the float precision model

```
python main_cifar10.py -a 'resnet20' --expr-name 'cifar10_resnet20_float'
```

2. Train the W4A4 model （Global Precison）

```
python main_cifar10.py -a 'resnet20' --expr-name 'cifar10_resnet20_w4a4_' --lr 5e-2 --wgt-target 4.0 --act-target 4.0 --quantize --pretrained [cifar10_resnet20_float_model_path]
```

3. Train the W2.0A2.0 model (Channel-wise Precision)

```
python main_cifar10.py -a 'resnet20' --expr-name 'cifar10_resnet20_w2.0a2.0' --lr 5e-2 --wgt-target 2.0 --act-target 2.0 --quantize --resume [cifar10_resnet20_W4A4_model_path]
```

## Results
Since this repo is totally reconstructed w.r.t original, there may be a little difference in results. We report this results as follows.

* ResNet20 on CIFAR10

|           | W4A4  | W2.0A2.0 |
| --------- | ----- | -------- |
| Original  | 92.45 | 91.69    |
| This repo | 92.71 | 92.13    |


## Cite
If this repo or paper is helpful, please cite,

```
@inproceedings{zhao2021distribution,
  title={Distribution-aware adaptive multi-bit quantization},
  author={Zhao, Sijie and Yue, Tao and Hu, Xuemei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9281--9290},
  year={2021}
}
```



