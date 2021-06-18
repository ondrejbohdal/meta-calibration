import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# soft accuracy:
# parts of the code are taken from https://github.com/wildltr/ptranking, specifically 
# https://github.com/wildltr/ptranking/blob/master/ptranking/base/neural_utils.py

# soft binning:
# we have used parts of the code from https://github.com/wOOL/DNDT/blob/master/pytorch/demo.ipynb


class Robust_Sigmoid(torch.autograd.Function):
    ''' Aiming for a stable sigmoid operator with specified sigma '''

    @staticmethod
    def forward(ctx, input, sigma=1.0, gpu=False):
        '''
        :param ctx:
        :param input: the input tensor
        :param sigma: the scaling constant
        :return:
        '''
        x = input if 1.0 == sigma else sigma * input

        torch_half = torch.cuda.FloatTensor(
            [0.5]) if gpu else torch.FloatTensor([0.5])
        sigmoid_x_pos = torch.where(
            input > 0, 1./(1. + torch.exp(-x)), torch_half)

        exp_x = torch.exp(x)
        sigmoid_x = torch.where(input < 0, exp_x/(1.+exp_x), sigmoid_x_pos)

        grad = sigmoid_x * \
            (1. - sigmoid_x) if 1.0 == sigma else sigma * \
            sigmoid_x * (1. - sigmoid_x)
        ctx.save_for_backward(grad)

        return sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output: backpropagated gradients from upper module(s)
        :return:
        '''
        grad = ctx.saved_tensors[0]

        bg = grad_output * grad  # chain rule

        return bg, None, None


#- function: robust_sigmoid-#
robust_sigmoid = Robust_Sigmoid.apply


class DECE(nn.Module):
    """
    Computes DECE loss (differentiable expected calibration error).
    """

    def __init__(self, device, num_bins, t_a, t_b):
        super(DECE, self).__init__()
        self.device = device
        self.num_bins = num_bins
        self.t_a = t_a
        self.t_b = t_b

    def one_hot(self, indices, depth):
        """
        Returns a one-hot tensor.
        This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        Parameters:
        indices:  a (n_batch, m) Tensor or (m) Tensor.
        depth: a scalar. Represents the depth of the one hot dimension.
        Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
        """
        encoded_indicies = torch.zeros(
            indices.size() + torch.Size([depth])).to(device=self.device)
        index = indices.view(indices.size() + torch.Size([1]))
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)

        return encoded_indicies

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C

        # For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with
        target = target.view(-1)

        predicted_probs = F.softmax(input, dim=1)

        cut_points = torch.linspace(0, 1, self.num_bins + 1)[:-1].to(device=self.device)
        W = torch.reshape(torch.linspace(1.0, self.num_bins, self.num_bins).to(device=self.device), [1, -1])
        b = torch.cumsum(-cut_points, 0)

        confidences = torch.max(predicted_probs, dim=1, keepdim=True)[0]
        h = torch.matmul(confidences, W) + b
        h = h / self.t_b

        bin_probs = F.softmax(h, dim=1)

        # smoothen the probabilities to avoid zeros
        eps = 1e-6
        bin_probs = bin_probs + eps
        # normalize the probabilities to sum to one across bins
        bin_probs = bin_probs / (1.0 + (self.num_bins + 1) * eps)

        # calculate bin confidences
        bin_confs = torch.div(bin_probs.transpose(0, 1).matmul(confidences).view(-1),
                              torch.sum(bin_probs, dim=0))
        # all-pairs approach
        batch_pred_diffs = torch.unsqueeze(predicted_probs, dim=2) - torch.unsqueeze(predicted_probs, dim=1)
        # computing pairwise differences, i.e., Sij or Sxy
        if str(self.device) == 'cpu':
            gpu = False
        else:
            gpu = True
        # using {-1.0*} may lead to a poor performance when compared with the above way;
        batch_indicators = robust_sigmoid(torch.transpose(batch_pred_diffs, dim0=1, dim1=2), self.t_a, gpu)

        # get approximated rank positions, i.e., hat_pi(x)
        ranks_all = torch.sum(batch_indicators, dim=2) + 0.5
        # the ranks go from 1 to C, with 1 being the best rank
        true_ranks = ranks_all[torch.arange(ranks_all.size(0)), target]
        accs = F.relu(2.0 - true_ranks)
        bin_accs = torch.div(bin_probs.transpose(0, 1).matmul(accs).view(-1),
                                torch.sum(bin_probs, dim=0))

        # calculate overall ECE for the whole batch
        ece = torch.sum(torch.sum(bin_probs, dim=0) * torch.abs(bin_accs - bin_confs) / bin_probs.shape[0], dim=0)
        return ece
