########## RESNET18 ##################

# CE
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss cross_entropy \
--save-path ../Models/ \
--exp_name resnet18_cifar100_cross_entropy

# Brier Loss
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss brier_score \
--save-path ../Models/ \
--exp_name resnet18_cifar100_brier_score

# MMCE
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss mmce_weighted --lamda 2.0 \
--save-path ../Models/ \
--exp_name resnet18_cifar100_mmce \

# Focal loss with fixed gamma 3 (FL-3)
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss focal_loss --gamma 3.0 \
--save-path ../Models/ \
--exp_name resnet18_cifar100_focal_loss

# Focal loss with sample dependent gamma 5,3 (FLSD-53)
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path ../Models/ \
--exp_name resnet18_cifar100_flsd

# Label smoothing 0.05
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss cross_entropy \
--label_smoothing 0.05 \
--save-path ../Models/ \
--exp_name resnet18_cifar100_ls

# Meta-Calibration
python ../train.py \
--dataset cifar100 \
--model resnet18 \
--loss cross_entropy \
--save-path ../Models/ \
--exp_name resnet18_cifar100_meta_calibration \
--meta_calibration non_uniform_label_smoothing