########## RESNET18 ##################

# CE
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_cross_entropy_best.model \
--exp_name resnet18_cifar10_cross_entropy \
>> resnet18_cifar10_cross_entropy.txt

# Brier Loss
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_brier_score_best.model \
--exp_name resnet18_cifar10_brier_score \
>> resnet18_cifar10_brier_score.txt

# MMCE
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_mmce_best.model \
--exp_name resnet18_cifar10_mmce \
>> resnet18_cifar10_mmce.txt

# Focal loss with fixed gamma 3 (FL-3)
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_focal_loss_best.model \
--exp_name resnet18_cifar10_focal_loss \
>> resnet18_cifar10_focal_loss.txt

# Focal loss with sample dependent gamma 5,3 (FLSD-53)
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_flsd_best.model \
--exp_name resnet18_cifar10_flsd \
>> resnet18_cifar10_flsd.txt

# Label smoothing 0.05
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_ls_best.model \
--exp_name resnet18_cifar10_ls \
>> resnet18_cifar10_ls.txt

# Meta-Calibration
python ../evaluate.py \
--dataset cifar10 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar10_meta_calibration_best.model \
--exp_name resnet18_cifar10_meta_calibration \
>> resnet18_cifar10_meta_calibration.txt