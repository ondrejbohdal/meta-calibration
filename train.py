"""
Script for training models.
"""

from torch import optim
import torch
import torch.nn as nn
import torch.utils.data
import argparse
import torch.backends.cudnn as cudnn
import random
import json
import sys
import os
import time
import tqdm

# Import dataloaders
import Data.cifar10 as cifar10
import Data.cifar100 as cifar100

# Import network models
import Net.resnet
from Net.resnet import resnet18, resnet50, resnet110
from Net.wide_resnet import wide_resnet_cifar

# Import loss functions
from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
from Losses.loss import dece

# Import train and validation utilities
from train_utils import train_single_epoch, test_single_epoch

# Import validation metrics
from Metrics.metrics import test_classification_net

dataset_num_classes = {"cifar10": 10, "cifar100": 100}

dataset_loader = {"cifar10": cifar10, "cifar100": cifar100}

models = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet110": resnet110,
    "wide_resnet": wide_resnet_cifar,
}


def create_json_experiment_log(json_experiment_log_file_name):
    experiment_summary_dict = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_ece": [],
        "val_err": [],
        "val_ece": [],
        "time": [],
        "learn_reg_vals": [],
    }
    with open(json_experiment_log_file_name, "w") as f:
        json.dump(experiment_summary_dict, fp=f)


def update_json_experiment_log_dict(
    experiment_update_dict, json_experiment_log_file_name
):
    with open(json_experiment_log_file_name, "r") as f:
        summary_dict = json.load(fp=f)

    for key in experiment_update_dict:
        if key not in summary_dict:
            summary_dict[key] = []
        summary_dict[key].append(experiment_update_dict[key])

    with open(json_experiment_log_file_name, "w") as f:
        json.dump(summary_dict, fp=f)


def parseArgs():
    default_dataset = "cifar10"
    dataset_root = "./"
    train_batch_size = 128
    test_batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    loss = "cross_entropy"
    gamma = 1.0
    gamma2 = 1.0
    gamma3 = 1.0
    lamda = 1.0
    weight_decay = 5e-4
    log_interval = 50
    save_interval = 50
    save_loc = "./"
    model_name = None
    saved_model_name = "resnet18_cross_entropy_350.model"
    load_loc = "./"
    model = "resnet18"
    exp_name = "resnet18_cross_entropy"
    epoch = 350
    first_milestone = 150  # Milestone for change in lr
    second_milestone = 250  # Milestone for change in lr
    gamma_schedule_step1 = 100
    gamma_schedule_step2 = 250

    parser = argparse.ArgumentParser(
        description="Training for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        dest="dataset",
        help="dataset to train on",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=dataset_root,
        dest="dataset_root",
        help="root path of the dataset",
    )
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument(
        "--load", action="store_true", dest="load", help="Load from pretrained model"
    )
    parser.set_defaults(load=False)
    parser.add_argument(
        "-b",
        type=int,
        default=train_batch_size,
        dest="train_batch_size",
        help="Batch size",
    )
    parser.add_argument(
        "-tb",
        type=int,
        default=test_batch_size,
        dest="test_batch_size",
        help="Test Batch size",
    )
    parser.add_argument(
        "-e", type=int, default=epoch, dest="epoch", help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=learning_rate,
        dest="learning_rate",
        help="Learning rate",
    )
    parser.add_argument(
        "--mom", type=float, default=momentum, dest="momentum", help="Momentum"
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        dest="nesterov",
        help="Whether to use nesterov momentum in SGD",
    )
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        "--decay",
        type=float,
        default=weight_decay,
        dest="weight_decay",
        help="Weight Decay",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default=optimiser,
        dest="optimiser",
        help="Choice of optimisation algorithm",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default=loss,
        dest="loss_function",
        help="Loss function to be used for training",
    )
    parser.add_argument(
        "--loss-mean",
        action="store_true",
        dest="loss_mean",
        help="whether to take mean of loss instead of sum to train",
    )
    parser.set_defaults(loss_mean=False)
    parser.add_argument(
        "--gamma",
        type=float,
        default=gamma,
        dest="gamma",
        help="Gamma for focal components",
    )
    parser.add_argument(
        "--gamma2",
        type=float,
        default=gamma2,
        dest="gamma2",
        help="Gamma for different focal components",
    )
    parser.add_argument(
        "--gamma3",
        type=float,
        default=gamma3,
        dest="gamma3",
        help="Gamma for different focal components",
    )
    parser.add_argument(
        "--lamda", type=float, default=lamda, dest="lamda", help="Regularization factor"
    )
    parser.add_argument(
        "--gamma-schedule",
        type=int,
        default=0,
        dest="gamma_schedule",
        help="Schedule gamma or not",
    )
    parser.add_argument(
        "--gamma-schedule-step1",
        type=int,
        default=gamma_schedule_step1,
        dest="gamma_schedule_step1",
        help="1st step for gamma schedule",
    )
    parser.add_argument(
        "--gamma-schedule-step2",
        type=int,
        default=gamma_schedule_step2,
        dest="gamma_schedule_step2",
        help="2nd step for gamma schedule",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        dest="label_smoothing",
        help="value to use for label smoothing when meta-training",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=log_interval,
        dest="log_interval",
        help="Log Interval on Terminal",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=save_interval,
        dest="save_interval",
        help="Save Interval on Terminal",
    )
    parser.add_argument(
        "--saved_model_name",
        type=str,
        default=saved_model_name,
        dest="saved_model_name",
        help="file name of the pre-trained model",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=save_loc,
        dest="save_loc",
        help="Path to export the model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=model_name,
        dest="model_name",
        help="name of the model",
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=load_loc,
        dest="load_loc",
        help="Path to load the model from",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=exp_name,
        dest="exp_name",
        help="name of the experiment",
    )

    parser.add_argument(
        "--model", type=str, default=model, dest="model", help="Model to train"
    )
    parser.add_argument(
        "--first-milestone",
        type=int,
        default=first_milestone,
        dest="first_milestone",
        help="First milestone to change lr",
    )
    parser.add_argument(
        "--second-milestone",
        type=int,
        default=second_milestone,
        dest="second_milestone",
        help="Second milestone to change lr",
    )

    parser.add_argument(
        "--meta_calibration",
        type=str,
        default="none",
        dest="meta_calibration",
        help="whether to use meta-calibration and its type",
    )
    parser.add_argument(
        "--meta_val_size",
        type=float,
        default=1.0,
        dest="meta_val_size",
        help="how large meta val set should be - relative to val",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=15,
        dest="num_bins",
        help="Number of bins for DECE calculation",
    )
    parser.add_argument(
        "--t_a",
        type=float,
        default=100.0,
        dest="t_a",
        help="Temperature for soft accuracy",
    )
    parser.add_argument(
        "--t_b",
        type=float,
        default=0.01,
        dest="t_b",
        help="Temperature for soft binning",
    )

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(1)
    args = parseArgs()

    cuda = False
    if torch.cuda.is_available() and args.gpu:
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    if args.meta_calibration != "none":
        Net.resnet.ResNet.meta = True
    net = models[args.model](num_classes=num_classes)

    # Setting model name
    if args.model_name is None:
        args.model_name = args.model

    if args.gpu is True:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    start_epoch = 0
    num_epochs = args.epoch
    if args.load:
        net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))
        start_epoch = int(
            args.saved_model_name[
                args.saved_model_name.rfind("_")
                + 1 : args.saved_model_name.rfind(".model")
            ]
        )

    if args.optimiser == "sgd":
        opt_params = net.parameters()
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        opt_params = net.parameters()
        optimizer = optim.Adam(
            opt_params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
    )

    if args.gpu is True:
        net_fc = net.module.fc
    else:
        net_fc = net.fc

    if args.meta_calibration == "learnable_l2":
        learnable_regularization = [
            nn.Parameter(torch.zeros_like(e)) for e in net_fc.parameters()
        ]
        meta_optimizer = torch.optim.Adam(learnable_regularization)
    elif args.meta_calibration == "scalar_label_smoothing":
        learnable_regularization = [nn.Parameter(torch.tensor(0.0).to(device))]
        meta_optimizer = torch.optim.Adam(learnable_regularization)
    elif args.meta_calibration == "vector_label_smoothing":
        learnable_regularization = [nn.Parameter(torch.zeros(num_classes).to(device))]
        meta_optimizer = torch.optim.Adam(learnable_regularization)
    elif args.meta_calibration == "non_uniform_label_smoothing":
        # the first item introduces non-uniformity in the label smoothing
        # the second item describes the total amount of label smoothing
        learnable_regularization = [
            nn.Parameter(torch.zeros(num_classes, num_classes).to(device)),
            nn.Parameter(torch.zeros(num_classes).to(device)),
        ]
        meta_optimizer = torch.optim.Adam(learnable_regularization)
    else:
        learnable_regularization = None
        meta_optimizer = None

    if args.meta_calibration != "none":
        train_loader, val_loader, meta_loader = dataset_loader[
            args.dataset
        ].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu,
            meta_val=True,
            meta_val_size=args.meta_val_size,
        )
    else:
        train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu,
        )

    test_loader = dataset_loader[args.dataset].get_test_loader(
        batch_size=args.test_batch_size, pin_memory=args.gpu
    )

    training_set_loss = {}
    val_set_loss = {}
    test_set_loss = {}
    val_set_err = {}

    for epoch in range(0, start_epoch):
        scheduler.step()

    # set up recording the statistics
    json_experiment_log_file_name = os.path.join("Experiments", args.exp_name + ".json")
    start_time = time.time()
    create_json_experiment_log(json_experiment_log_file_name)

    best_val_acc = 0
    with tqdm.tqdm(total=num_epochs) as pbar_epochs:
        for epoch in range(start_epoch, num_epochs):
            scheduler.step()
            if args.loss_function == "focal_loss" and args.gamma_schedule == 1:
                if epoch < args.gamma_schedule_step1:
                    gamma = args.gamma
                elif (
                    epoch >= args.gamma_schedule_step1
                    and epoch < args.gamma_schedule_step2
                ):
                    gamma = args.gamma2
                else:
                    gamma = args.gamma3
            else:
                gamma = args.gamma

            if args.meta_calibration != "none":
                meta_loader_copy = meta_loader
            else:
                meta_loader_copy = val_loader
            train_loss, train_ece = train_single_epoch(
                epoch,
                net,
                train_loader,
                meta_loader_copy,
                optimizer,
                meta_optimizer,
                learnable_regularization,
                device,
                args,
                loss_function=args.loss_function,
                gamma=gamma,
                lamda=args.lamda,
                loss_mean=args.loss_mean,
            )
            val_loss = test_single_epoch(
                epoch,
                net,
                val_loader,
                device,
                args,
                loss_function=args.loss_function,
                gamma=gamma,
                lamda=args.lamda,
            )
            test_loss = test_single_epoch(
                epoch,
                net,
                val_loader,
                device,
                args,
                loss_function=args.loss_function,
                gamma=gamma,
                lamda=args.lamda,
            )
            _, val_acc, _, _, _, val_ece = test_classification_net(
                net, val_loader, device, return_ece=True
            )

            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss
            val_set_err[epoch] = 1 - val_acc

            experiment_update_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_ece": train_ece,
                "val_err": 1 - val_acc,
                "val_ece": val_ece,
            }
            if args.meta_calibration == "scalar_label_smoothing":
                experiment_update_dict["learn_reg_vals"] = learnable_regularization[
                    -1
                ].item()
            if args.meta_calibration == "vector_label_smoothing":
                experiment_update_dict["learn_reg_vals"] = learnable_regularization[
                    -1
                ].tolist()
            if args.meta_calibration == "non_uniform_label_smoothing":
                # only store the overall scaling to avoid too large json file
                experiment_update_dict["learn_reg_vals"] = learnable_regularization[
                    1
                ].tolist()

            update_json_experiment_log_dict(
                experiment_update_dict, json_experiment_log_file_name
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("New best error: %.4f" % (1 - best_val_acc))
                save_name = args.save_loc + args.exp_name + "_best.model"
                torch.save(net.state_dict(), save_name)

            if (epoch + 1) % args.save_interval == 0:
                save_name = (
                    args.save_loc + args.exp_name + "_" + str(epoch + 1) + ".model"
                )
                torch.save(net.state_dict(), save_name)

            pbar_epochs.update(1)

    experiment_update_dict = {"time": time.time() - start_time}
    update_json_experiment_log_dict(
        experiment_update_dict, json_experiment_log_file_name
    )
