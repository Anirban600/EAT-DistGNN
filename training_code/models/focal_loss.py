import torch.nn as nn
import torch.nn.functional as F
import torch as th


class FocalLoss(nn.Module):
    def __init__(self, n_classes, gamma=0, alpha=1, size_average=True, multilabel=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
        self.n_classes = n_classes
        self.multilabel= multilabel
        
    def forward(self, logits, labels):
        """
        calculates loss
        logits: batch_size * labels_length
        labels: batch_size
        """
        # print("First",labels[0])
        assert(logits.size(0) == labels.size(0))

        # transpose labels into labels onehot
        if self.multilabel:
            p = th.sigmoid(logits)
            sub_pt = th.clamp(1 - p, 0.00000001, 1.0)
            fl = -(labels*((sub_pt)**self.gamma)*th.log(p+0.00000001) + (1-labels)*(p**self.gamma)*th.log(sub_pt))
        else:
            label_onehot = F.one_hot(labels, num_classes=self.n_classes)
            p = F.softmax(logits, dim=1)
            sub_pt = th.clamp(1 - p, 0.00000001, 1.0)
            fl = -self.alpha * label_onehot * \
                ((sub_pt)**self.gamma) * th.log(p+0.00000001)
        if self.size_average:
            return fl.sum()/labels.size(0)
        else:
            return fl.sum()

