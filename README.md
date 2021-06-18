# Meta-Calibration: Meta-Learning of Model Calibration Using Differentiable Expected Calibration Error

[[Paper]](https://arxiv.org/abs/2106.09613)

Calibration of neural networks is a topical problem that is becoming increasingly important for real-world use of neural networks. The problem is especially noticeable when using modern neural networks, for which there is significant difference between the model confidence and the confidence it should have. Various strategies have been successfully proposed, yet there is more space for improvements. We propose a novel approach that introduces a differentiable metric for expected calibration error and successfully uses it as an objective for meta-learning, achieving competitive results with state-of-the-art approaches. Our approach presents a new direction of using meta-learning to directly optimize model calibration, which we believe will inspire further work in this promising and new direction.

Our implementation extends the [implementation](https://github.com/torrvision/focal_calibration) for paper [*Calibrating Deep Neural Networks using Focal Loss*](https://arxiv.org/abs/2002.09437) from Mukhoti et al. You can find further useful information there.


<p align="center"><img src='DECEandECEcorrelations.png' width=700></p>

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
The approach is implemented in PyTorch and its dependacies are listed in [environment.yml](environment.yml).

### Datasets
CIFAR-10 and CIFAR-100 datasets will be downloaded automatically.

## Experiments

You can train and evaluate a model with meta-calibration using the following commands:
```
python train.py --dataset cifar10 --model resnet18 --loss cross_entropy --save-path Models/ --exp_name rn18_c10_meta_calibration --meta_calibration

python evaluate.py --dataset cifar10 --model resnet18 --save-path Models/ --saved_model_name rn18_c10_meta_calibration_best.model --exp_name rn18_c10_meta_calibration
``` 

## Citation

If you find this useful for your research, please consider citing:
 ```
 @article{bohdal2021meta-calibration,
   title={Meta-Calibration: Meta-Learning of Model Calibration Using Differentiable Expected Calibration Error},
   author={Bohdal, Ondrej and Yang, Yongxin and Hospedales, Timothy},
   journal={arXiv preprint arXiv:2106.09613},
   year={2021}
}
 ```

## Acknowledgments

This work was supported in part by the EPSRC Centre for Doctoral Training in Data Science, funded by the UK Engineering and Physical Sciences Research Council (grant EP/L016427/1) and the University of Edinburgh.