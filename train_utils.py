'''
This module contains methods for training models with different loss functions.
'''

from itertools import cycle

import torch
from torch import nn
from torch.nn import functional as F

from Losses.loss import (brier_score, cross_entropy, dece, focal_loss,
                         focal_loss_adaptive, mmce, mmce_weighted)
from Metrics.metrics import ECELoss

loss_function_dict = {
    'cross_entropy': cross_entropy,
    'focal_loss': focal_loss,
    'focal_loss_adaptive': focal_loss_adaptive,
    'mmce': mmce,
    'mmce_weighted': mmce_weighted,
    'brier_score': brier_score,
    'dece': dece
}


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def calculate_loss(logits, labels, learnable_regularization, loss_function,
                   gamma, lamda, device, model_fc, args):
    if args.meta_calibration == 'scalar_label_smoothing' or args.meta_calibration == 'vector_label_smoothing':
        if args.dataset == 'cifar10':
            num_classes = 10
        elif args.dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError('Dataset ' + args.dataset + ' is not implemented yet.')

        if args.meta_calibration == 'scalar_label_smoothing':
            oh_labels = one_hot(labels, num_classes, device)
            soft_labels = create_smooth_labels(
                oh_labels, learnable_regularization[0], num_classes)
        else:
            soft_labels = create_class_smooth_labels(
                labels, learnable_regularization[0], num_classes, device)

        # soft_cross_entropy does sum reduction
        loss = soft_cross_entropy(logits, soft_labels)
    elif args.meta_calibration == 'learnable_l2':
        loss = loss_function_dict[loss_function](
            logits, labels, gamma=gamma, lamda=lamda, device=device)
        for param, reg_w in zip(model_fc.parameters(), learnable_regularization):
            loss += torch.sum(reg_w * (param ** 2))
    else:
        loss = loss_function_dict[loss_function](
            logits, labels, gamma=gamma, lamda=lamda, device=device)
    return loss


def one_hot(indices, depth, device):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(
        indices.size() + torch.Size([depth])).to(device=device)
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def create_smooth_labels(labels, label_smoothing, num_classes):
    labels = labels * (1.0 - label_smoothing)
    labels = labels + label_smoothing / num_classes
    return labels


def create_class_smooth_labels(labels, label_smoothing, num_classes, device):
    """A method for calculating soft labels with smoothing specific to the class"""
    targets = []
    for target in labels:
        soft_target = torch.zeros(num_classes).to(device=device)
        soft_target[target] = (1.0 - label_smoothing[target])
        soft_target += label_smoothing[target] / num_classes
        targets.append(soft_target)
    return torch.stack(targets)


def soft_cross_entropy(pred, soft_targets):
    """A method for calculating cross entropy with soft targets"""
    logsoftmax = nn.LogSoftmax()
    # we use sum reduction
    return torch.sum(torch.sum(- soft_targets * logsoftmax(pred), 1))


def train_single_epoch(epoch,
                       model,
                       train_loader,
                       val_loader,
                       optimizer,
                       meta_optimizer,
                       learnable_regularization,
                       device,
                       args,
                       loss_function='cross_entropy',
                       gamma=1.0,
                       lamda=1.0,
                       loss_mean=False):
    '''
    Util method for training a model for a single epoch.
    '''
    log_interval = 10
    model.train()
    train_loss = 0
    train_ece = 0
    ece_criterion = ECELoss().to(device)
    num_samples = 0
    if args.meta_calibration != "none":
        loaders = zip(train_loader, cycle(val_loader))
    else:
        loaders = train_loader
    current_LR = get_learning_rate(optimizer)[0]

    if args.gpu is True:
        model_fc = model.module.fc
    else:
        model_fc = model.fc

    for batch_idx, batch in enumerate(loaders):
        if args.meta_calibration != "none":
            ((data, labels), (data_val, labels_val)) = batch
            data_val = data_val.to(device)
            labels_val = labels_val.to(device)
        else:
            (data, labels) = batch
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if args.meta_calibration != "none":
            # simulate an update - but only using the classifier - the rest will appear frozen
            fast_parameters = list(model_fc.parameters())
            for weight in model_fc.parameters():
                weight.fast = None
            optimizer.zero_grad()

            logits = model(data)
            loss = calculate_loss(
                logits, labels, learnable_regularization, loss_function, gamma, lamda, device, model_fc, args)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for k, weight in enumerate(model_fc.parameters()):
                if weight.fast is None:
                    weight.fast = weight - current_LR * grad[k]
                else:
                    weight.fast = weight.fast - current_LR * grad[k]
                fast_parameters.append(weight.fast)

            # outer loop
            # this will use the fast weights because they have not been reset to None
            logits_val = model(data_val)

            meta_loss = loss_function_dict['dece'](logits_val, labels_val, device=device,
                                                   num_bins=args.num_bins, t_a=args.t_a, t_b=args.t_b)
            meta_optimizer.zero_grad()
            meta_loss.backward(retain_graph=True)
            meta_optimizer.step()

            if args.meta_calibration == 'scalar_label_smoothing' or args.meta_calibration == 'vector_label_smoothing':
                learnable_regularization[0].data = torch.clamp(
                    learnable_regularization[0], min=0.0, max=0.5).data

            # reset the fast weights to None
            for weight in model_fc.parameters():
                weight.fast = None

            # standard update of the model is done afterwards
            loss = calculate_loss(
                logits, labels, learnable_regularization, loss_function, gamma, lamda, device, model_fc, args)
            train_ece += len(data) * ece_criterion(logits, labels).item()
        else:
            logits = model(data)
            if ('mmce' in loss_function):
                loss = (len(data) * loss_function_dict[loss_function](
                    logits, labels, gamma=gamma, lamda=lamda, device=device))
                train_ece += len(data) * ece_criterion(logits, labels).item()
            else:
                if args.label_smoothing != 0.0:
                    if args.dataset == 'cifar10':
                        num_classes = 10
                    elif args.dataset == 'cifar100':
                        num_classes = 100
                    else:
                        raise ValueError('Unknown dataset')
                    oh_labels = one_hot(labels, num_classes, device)
                    soft_labels = create_smooth_labels(
                        oh_labels, args.label_smoothing, num_classes)
                    # soft_cross_entropy does sum reduction
                    loss = soft_cross_entropy(logits, soft_labels)
                else:
                    loss = loss_function_dict[loss_function](
                        logits, labels, gamma=gamma, lamda=lamda, device=device)
                train_ece += len(data) * ece_criterion(logits, labels).item()

        if loss_mean:
            loss = loss / len(data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples, train_ece / num_samples


def test_single_epoch(epoch,
                      model,
                      test_val_loader,
                      device,
                      args,
                      loss_function='cross_entropy',
                      gamma=1.0,
                      lamda=1.0):
    '''
    Util method for testing a model for a single epoch.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            if ('mmce' in loss_function):
                loss += (len(data) * loss_function_dict[loss_function](
                    logits, labels, gamma=gamma, lamda=lamda, device=device).item())
            else:
                loss += loss_function_dict[loss_function](
                    logits, labels, gamma=gamma, lamda=lamda, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples
