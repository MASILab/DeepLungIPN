import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import math


class LossPool(object):
    '''
    https://pytorch.org/docs/0.3.0/nn.html#loss-functions
    please add all these loss to here.
    self.pred.shape = (N x D)
    self.target.shape = (N,)
    '''
    def __init__(self, pred, target, cfig, feat = None, scan_t = None, diag_t = None, loss_name = None):
        self.loss_name = loss_name
        self.pred = pred
        self.target = target
        self.cfig = cfig
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scan_t = scan_t
        self.diag_t = diag_t
        self.feat = feat
        
    def get_loss(self):
        method = getattr(self, self.loss_name)
        return method()
    
    def nll_loss(self):
        print ('===========using   nll_loss================')
        return nn.NLLLoss()(self.pred, self.target)
    
    def cross_entropy_loss(self):
        return nn.CrossEntropyLoss()(self.pred, self.target)
    
    def bi_ce_loss(self):
        return nn.BCELoss()(self.pred, self.target)

    def weight_bceloss(self):
        loss = weighted_binary_cross_entropy(self.pred, self.target, self.cfig['bce_weight'])
        return loss

    def focal_loss(self):
        print('===========using  focal_loss================')
        #alpha = torch.from_numpy(np.array(self.cfig['focal_alpha'], dtype = np.float32))
        #gamma = torch.from_numpy(np.array(self.cfig['focal_gamma'], dtype = np.float32))
        #print (alpha)
        focal = FocalLoss(class_num = 2)
        loss = focal(self.pred, self.target)
        return loss

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs, targets):
        
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print('class_mask',class_mask)  # like one-hot 
        #print ('self.alpha ', self.alpha)
        #print (inputs.is_cuda, self.alpha.is_cuda)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha = self.alpha[ids.data.view(-1)]
        #print (ids.data.view(-1))
        #print (alpha, self.alpha)

        probs = (P * class_mask).sum(1).view(-1, 1)
        #print ('probs', probs)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        #print (alpha.shape, log_p.shape)
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p # self.gamma.float().to(self.device)
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
class BFocalLoss(nn.Module):
    #https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(BFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def weighted_binary_cross_entropy(output, target, weights=None):
        #print (target)
        #print (output)
        
    ones_device = torch.ones(output.shape).cuda()
    
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target.cuda().float() * torch.log(torch.min(0.999 * ones_device, torch.max(output, 0.00001 * ones_device)))) + weights[0] * ((1. - target.cuda().float()) * torch.log(1.000 - output))
    else:
        loss = target.cuda() * torch.log(torch.min(0.999 * ones_device, torch.max(output, 0.00001 * ones_device))) + (1. - target.cuda()) * torch.log(1.0 - output)

    return torch.neg(torch.mean(loss))


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        torch.manual_seed(666)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc()
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        #print ('---------batch_size_tensor---------', batch_size_tensor)
        #print (feat, label, self.centers)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss

class CenterlossFunc(nn.Module):
    def __init__(self):
        super(CenterlossFunc, self).__init__()
    def forward(self, feature, label, centers, batch_size):
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size