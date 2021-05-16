import torch
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['KL_divergence']

def KL_divergence(logits_p, logits_q):
    # p = softmax(logits_p)
    # q = softmax(logits_q)
    # KL(p||q)
    # suppose that p/q is in shape of [bs, num_classes]

    #p = F.softmax(logits_p, dim=1)
    #q = F.softmax(logits_q, dim=1)

    shape = list(logits_p.size())
    _shape = list(logits_q.size())
    assert shape == _shape
    #print(shape)
    num_classes = shape[1]
    epsilon = 1e-8
    _p = (logits_p + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    _q = (logits_q + epsilon * Variable(torch.ones(*shape).cuda())) / (1.0 + num_classes * epsilon)
    return torch.mean(torch.sum(_p * torch.log(_p / _q), 1))
