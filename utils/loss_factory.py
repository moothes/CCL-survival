import torch
import math 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(hazards, S, Y, c, alpha=0., eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def __call__(self, out, gt, alpha=None):
        loss = 0
        for haz, s in zip(out['hazards'], out['S']):
            loss += nll_loss(haz, s, gt['label'], gt['c'], alpha=self.alpha)
        return loss

class CohortLoss():
    def __call__(self, out, gt, temperature=2):
        loss = 0
        if 'cohort' in out.keys():
            alpha = 10
            indiv, origs = out['decompose']
            cohort, c = out['cohort']
            
            mask = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1]]).cuda()
            indiv_know = indiv.view(4, 1, -1) # common, synergy, g_spec, p_spec
            orig = torch.cat(origs, dim=1).detach() # gene, path
            sim = F.cosine_similarity(indiv_know, orig, dim=-1)
            intra_loss = torch.mean(torch.abs(sim) * (1 - mask) - mask * sim) + 1
            
            if c is None:
                return 0
            if int(c) == 0:
                neg_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort) if int(gt['label']) != j], dim=0).detach()
                pos_feat = cohort[int(gt['label'])][:-1].detach()
            else:
                if int(gt['label']) != 0:
                    neg_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort) if int(gt['label']) > j], dim=0).detach()
                    pos_feat = torch.cat([feat.detach() for j, feat in enumerate(cohort) if int(gt['label']) <= j], dim=0).detach()
                else:
                    return intra_loss.mean()
                    
            if neg_feat.shape[0] < 1 or pos_feat.shape[0] < 1:
                inter_loss = 0
            else:
                neg_dis = indiv_know.squeeze(1) * neg_feat / temperature
                pos_dis = indiv_know.squeeze(1) * pos_feat / temperature
                inter_loss = -torch.log(torch.exp(pos_dis).mean() / (torch.exp(pos_dis).mean() + torch.exp(neg_dis).mean() + 1e-10)) * 1
            
            loss = intra_loss.mean() + inter_loss
        return loss 

loss_dict = {'nllsurv': NLLSurvLoss(), 'cohort': CohortLoss()}

class Loss_factory(nn.Module):
    def __init__(self, args):
        super(Loss_factory, self).__init__()
        loss_item = args.loss.split(',')
        self.loss_collection = {}
        for loss_im in loss_item:
            tags = loss_im.split('_')
            if len(tags) == 2:
                self.loss_collection[tags[0]] = float(tags[1])
            else:
                self.loss_collection[tags[0]] = 1.
        
    def forward(self, preds, target):
        loss_sum = 0
        ldict = {}
        for loss_name, weight in self.loss_collection.items():
            loss = loss_dict[loss_name](preds, target) * weight
            ldict[loss_name] = loss
            loss_sum += loss
        return loss_sum, ldict
