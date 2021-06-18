########## RESNET18 ##################

# CE
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_cross_entropy_best.model \
--exp_name resnet18_cifar100_cross_entropy \
>> resnet18_cifar100_cross_entropy.txt

# Brier Loss
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_brier_score_best.model \
--exp_name resnet18_cifar100_brier_score \
>> resnet18_cifar100_brier_score.txt

# MMCE
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_mmce_best.model \
--exp_name resnet18_cifar100_mmce \
>> resnet18_cifar100_mmce.txt

# Focal loss with fixed gamma 3 (FL-3)
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_focal_loss_best.model \
--exp_name resnet18_cifar100_focal_loss \
>> resnet18_cifar100_focal_loss.txt

# Focal loss with sample dependent gamma 5,3 (FLSD-53)
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_flsd_best.model \
--exp_name resnet18_cifar100_flsd \
>> resnet18_cifar100_flsd.txt

# Label smoothing 0.05
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_ls_best.model \
--exp_name resnet18_cifar100_ls \
>> resnet18_cifar100_ls.txt

# Meta-Calibration
python ../evaluate.py \
--dataset cifar100 \
--model resnet18 \
--save-path ../Models/ \
--saved_model_name resnet18_cifar100_meta_calibration_best.model \
--exp_name resnet18_cifar100_meta_calibration \
>> resnet18_cifar100_meta_calibration.txt