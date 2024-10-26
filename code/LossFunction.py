import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # https://pytorch.org/docs/1.2.0/nn.html?highlight=marginrankingloss#torch.nn.MarginRankingLoss
        # 计算两个张量之间的相似度，两张量之间的距离>margin，loss 为正，否则loss 为 0
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)	# batch_size
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


# implementation of triplet center loss
# see https://github.com/xlliu7/Shrec2018_TripletCenterLoss.pytorch/blob/master/misc/custom_loss.py  # [batch,dim]
class TripletCenterLoss(nn.Module):
    # note the device
    def __init__(self, margin=5.0, num_classes=2, num_dim=1024):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, num_dim).cuda())  # random initialize as parameters

    def forward(self, inputs, targets):
        # resize inputs, delete labels with -1
        # inputs = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2))
        # targets = targets.reshape(targets.size(0) * targets.size(1))
        # ignore_idx = targets != -1
        # inputs = inputs[ignore_idx]
        # targets = targets[ignore_idx]


        batch_size = inputs.size(0)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))  # [batch, dim]
        centers_batch = self.centers.gather(0, targets_expand)  # 取出相应index的embedding

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)  # [batch, batch, dim]
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)  # as above
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative (the furthest positive and nearest negative)
        # hard mining
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            # if dist[i][mask[i] == 0].numel() == 0:
            #     continue
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        # y_i = 1, means dist_an > dist_ap + margin will causes loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


# class TripletCenterLoss(nn.Module):
#     def __init__(self, margin=0, num_classes=2):
#         super(TripletCenterLoss, self).__init__() 
#         self.margin = margin 
#         self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
#         self.centers = nn.Parameter(torch.randn(num_classes,1024).cuda())
   
#     def forward(self, inputs, targets): 
#         batch_size = inputs.size(0) 
#         targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))
#         # targets_expand = targets.view(batch_size, 1)
#         # print(targets_expand.device,self.centers.device)
#         # self.centers.to(targets_expand.device)
#         # print(targets_expand.device,self.centers.device)
#         centers_batch = self.centers.gather(0, targets_expand) # centers batch 

#         # compute pairwise distances between input features and corresponding centers 
#         centers_batch_bz = torch.stack([centers_batch]*batch_size) 
#         inputs_bz = torch.stack([inputs]*batch_size).transpose(0, 1) 
#         dist = torch.sum((centers_batch_bz -inputs_bz)**2, 2).squeeze() 
#         dist = dist.clamp(min=1e-12).sqrt() # for numerical stability 

#         # for each anchor, find the hardest positive and negative 
#         mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
#         dist_ap, dist_an = [], [] 
#         for i in range(batch_size): # for each sample, we compute distance 
#             dist_ap.append(dist[i][mask[i]].max()) # mask[i]: positive samples of sample i
#             dist_an.append(dist[i][mask[i]==0].min()) # mask[i]==0: negative samples of sample i 

#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)

#         # generate a new label y
#         # compute ranking hinge loss 
#         y = dist_an.data.new() 
#         y.resize_as_(dist_an.data)
#         y.fill_(1)
#         y = Variable(y)
#         # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero 
#         loss = self.ranking_loss(dist_an, dist_ap, y)

#         prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0) # normalize data by batch size 
#         return loss, prec    