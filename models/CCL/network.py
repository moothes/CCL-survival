import torch
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
from .kmeans.kmeans_pytorch import kmeans
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings("ignore")

from .util import NystromAttention

custom_config = {'base'      : {'modal': 'multi',
                                'loss': 'nllsurv,cohort',
                                'sets': 'blca,brca,luad,ucec,gbmlgg',
                                'lr': 1e-3,
                                'optimizer': 'SGD',
                                'scheduler': 'None',
                                'num_epoch': 30,
                                'seed': 0,
                               },
                 'customized': {'num_cluster': {'type': int, 'default': 6},
                                'update_ratio': {'type': float, 'default': 0.1},
                                'bank_length': {'type': int, 'default': 10},
                               },
                }
                
def SNN_Block(dim1, dim2, dropout=0.15):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.SELU(), nn.AlphaDropout(p=dropout, inplace=False))
    
def MLP_Block(dim1, dim2, dropout=0.15):
    return nn.Sequential(nn.Linear(dim1, dim2), nn.LayerNorm(dim2), nn.ReLU())

def conv1d_Block(dim1, dim2, dropout=0.15):
    return nn.Sequential(nn.Conv1d(dim1, dim2, 1), nn.InstanceNorm1d(dim2), nn.ReLU())

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim//2,
            pinv_iterations=6,
            residual=True, 
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class Transformer(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer, self).__init__()
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        B = features.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class Specificity_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=64):
        super().__init__()
        self.conv = MLP_Block(dim, dim)
        
    def forward(self, feat):
        feat = self.conv(feat)
        return feat

class Interaction_Estimator(nn.Module):
    def __init__(self, feat_len=6, dim=64):
        super().__init__()
        self.geno_fc = MLP_Block(dim, dim)
        self.path_fc = MLP_Block(dim, dim)
        self.geno_atten = nn.Linear(dim, 1)
        self.path_atten = nn.Linear(dim, 1)
        
    def forward(self, gfeat, pfeat):        
        g_align = self.geno_fc(gfeat)
        p_align = self.path_fc(pfeat)
        atten = g_align.unsqueeze(3) * p_align.unsqueeze(2)
        geno_att = torch.sigmoid(self.geno_atten(atten)).squeeze(-1)
        path_att = torch.sigmoid(self.path_atten(atten.permute(0, 1, 3, 2))).squeeze(-1)
        interaction = p_align * path_att + g_align * geno_att
        return interaction

class Knowledge_Decomposition(nn.Module):
    def __init__(self, feat_len=6, feat_dim=64):
        super().__init__()
        self.geno_spec = Specificity_Estimator(feat_len, feat_dim)
        self.path_spec = Specificity_Estimator(feat_len, feat_dim)
        
        self.common_encoder = Interaction_Estimator(feat_len, feat_dim)
        self.synergy_encoder = Interaction_Estimator(feat_len, feat_dim)
        
    def forward(self, gfeat, pfeat):
        g_spec = self.geno_spec(gfeat)
        p_spec = self.path_spec(pfeat)
        common = self.common_encoder(pfeat, gfeat)
        synergy = self.synergy_encoder(pfeat, gfeat)
        return common, synergy, g_spec, p_spec

def Hungarian_Matching(centers, priors):
    cost = torch.cdist(centers, priors, p=1).detach().cpu()
    indices = linear_sum_assignment(cost)[-1]
    one_hot_targets = F.one_hot(torch.tensor(indices), centers.shape[0]).float().cuda()
    align_centers = torch.mm(one_hot_targets.T, centers)
    return align_centers

genomics_idx = {'blca':   [0, 94, 428, 949, 1417, 2913, 3392],
                'brca':   [0, 91, 444, 997, 1477, 3043, 3523],
                'gbmlgg': [0, 84, 398, 896, 1311, 2707, 3135],
                'luad':   [0, 89, 423, 957, 1428, 2938, 3420],
                'ucec':   [0, 3, 27, 48, 70, 135, 150],
                }

class CCL(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seed = args.seed
        self.feat_dim = 256
        self.n_classes = args.n_classes
        self.num_cluster = args.num_cluster
        self.bank_length = args.bank_length
        if args.extractor in ('uni', 'resnet50'):
            path_dim = 1024
        elif args.extractor in ('plip', 'conch'):
            path_dim = 512
        elif args.extractor in ('hipt', ):
            path_dim = 384

        # Genomic representation
        self.genomics_idx = genomics_idx[args.dataset]
        sig_networks = []
        for idx in range(len(self.genomics_idx)-1):
            fc_omic = SNN_Block(dim1=self.genomics_idx[idx+1] - self.genomics_idx[idx], dim2=256)
            sig_networks.append(nn.Sequential(fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)
        self.geno_fc = SNN_Block(dim1=256, dim2=self.feat_dim)
        self.geno_conv = conv1d_Block(6, 1)
         
        # pathology representation
        self.path_know_memory = nn.Parameter(torch.randn(1, self.num_cluster, path_dim), requires_grad=False).cuda()
        self.update_ratio = args.update_ratio
        self.path_fc = SNN_Block(dim1=path_dim, dim2=self.feat_dim)   
        self.path_conv = conv1d_Block(self.num_cluster, 1)     
                        
        # Cross-modal Knowledge Decoupling
        self.know_decompose = Knowledge_Decomposition(self.num_cluster, self.feat_dim)
        
        # Cohort Bank
        self.patient_bank = []
        for i in range(self.n_classes):
            pbank = nn.Parameter(torch.randn(0, 4, self.feat_dim), requires_grad=False).cuda()
            self.patient_bank.append(pbank)
        
        self.transformer = Transformer(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, self.n_classes)

    def forward(self, x_path, x_omic, phase, label=None, c=None, **kwargs):
        out_dict = {}
        geno_feat = torch.stack([self.genomics_fc[idx].forward(x_omic[..., length:self.genomics_idx[idx+1]]) for idx, length in enumerate(self.genomics_idx[:-1])], dim=1)
        geno_indiv = self.geno_conv(self.geno_fc(geno_feat))
        
        with torch.no_grad():
            cluster_ids_x, path_centers = kmeans(X=x_path[0], num_clusters=self.num_cluster, cluster_centers=self.path_know_memory.detach(), distance='euclidean', device=torch.device('cuda:0'), tqdm_flag=False, seed=self.seed)
        path_centers = Hungarian_Matching(path_centers.cuda(), self.path_know_memory[0]).unsqueeze(0)
        if phase == 'train':
            self.path_know_memory = (self.path_know_memory * (1 - self.update_ratio) + path_centers * self.update_ratio).detach()         
        path_indiv = self.path_conv(self.path_fc(path_centers))
        
        # components = common, synergy, geno_spec, path_spec
        indiv_components = self.know_decompose(path_indiv, geno_indiv)
        indiv_know = torch.cat(indiv_components, dim=1)
        if phase == 'train':
                
            if self.patient_bank[int(label)].shape[0] < self.bank_length:
                self.patient_bank[int(label)] = torch.cat([self.patient_bank[int(label)], indiv_know], dim=0)
            else:
                self.patient_bank[int(label)] = torch.cat([self.patient_bank[int(label)][1:], indiv_know], dim=0)
        
        fusion, _ = self.transformer(indiv_know)
        fuse_hazard = torch.sigmoid(self.classifier(fusion))
        fuse_S = torch.cumprod(1 - fuse_hazard, dim=1)
        
        # prediction
        out_dict['decompose'] = [indiv_know, [geno_indiv, path_indiv]]
        out_dict['cohort'] = [self.patient_bank, c]
        out_dict['hazards'] = [fuse_hazard]
        out_dict['S'] = [fuse_S]
        return out_dict
