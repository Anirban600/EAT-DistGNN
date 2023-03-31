import torch.nn as nn
import torch.nn.functional as F
import torch as th


class FocalLoss(nn.Module):
    def __init__(self, n_classes, gamma=0, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
        self.n_classes = n_classes

    def forward(self, logits, labels):
        """
        calculates loss
        logits: batch_size * labels_length
        labels: batch_size
        """
        # print(logits.size(),labels.size())
        assert(logits.size(0) == labels.size(0))

        # transpose labels into labels onehot
        label_onehot = F.one_hot(labels, num_classes=self.n_classes)

        # calculate log
        p = F.softmax(logits, dim=1)
        sub_pt = (1 - p)
        fl = -self.alpha * label_onehot * \
            (sub_pt)**self.gamma * th.log(p+0.000000001)
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
