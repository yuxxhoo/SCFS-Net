"""SCFS-Net: FSCA, CAFD and MHSR on shared multispectral features."""
import torch
from torch import nn
from .backbone import Backbone
from .clip_visual import QuickGELU
from .fusion import SCFSFusion

def init_bn(m):
    if isinstance(m,nn.BatchNorm1d):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

class SCFSAdapter(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.fusion=SCFSFusion(512,cfg)
    def forward(self,*features):
        if self.fusion.cfg.MODEL.SCFS.DETACH_BACKBONE:
            features=tuple(x.detach() for x in features)
        result=self.fusion(*features)
        if self.training:
            aux=self.fusion.get_auxiliary_loss() if self.fusion.cfg.MODEL.SCFS.USE_AUXILIARY_LOSS else result.new_zeros(())
            return result,aux
        return result

class SCFSNet(nn.Module):
    def __init__(self,num_classes,cfg,camera_num,view_num=0):
        super().__init__(); self.cfg=cfg; self.feat_dim=512
        self.backbone=Backbone(cfg,camera_num)
        self.collaboration=SCFSAdapter(cfg)
        self.direct=cfg.MODEL.DIRECT
        self.GLOBAL_LOCAL=cfg.MODEL.GLOBAL_LOCAL
        if self.GLOBAL_LOCAL:
            self.pool=nn.AdaptiveAvgPool1d(1)
            for modality in ('rgb','nir','tir'):
                setattr(self,modality+'_reduce',nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,512),QuickGELU()))
        for suffix,dim in ([('scfs',1536),('',1536)] if self.direct else [('scfs',1536),('r',512),('n',512),('t',512)]):
            tag='_'+suffix if suffix else ''
            classifier=nn.Linear(dim,num_classes,bias=False)
            nn.init.normal_(classifier.weight,std=.001)
            bn=nn.BatchNorm1d(dim); bn.bias.requires_grad_(False); bn.apply(init_bn)
            setattr(self,'classifier'+tag,classifier); setattr(self,'bottleneck'+tag,bn)
    def forward(self,x,label=None,cam_label=None,view_label=None,return_pattern=3,img_path=None):
        images=[x[k] for k in ('RGB','NI','TI')]
        if not self.training:
            missing=self.cfg.TEST.MISS
            images=[torch.zeros_like(v) if k in missing else v for k,v in zip(('r','n','t'),images)] if missing!='nothing' else images
        pairs=[self.backbone(v,cam_label,view_label) for v in images]
        tokens=[v[0] for v in pairs]; glob=[v[1] for v in pairs]
        if self.GLOBAL_LOCAL:
            glob=[getattr(self,k+'_reduce')(torch.cat((g,self.pool(t.transpose(1,2)).squeeze(-1)),dim=-1)) for k,t,g in zip(('rgb','nir','tir'),tokens,glob)]
        result=self.collaboration(*(tokens+glob))
        ori=torch.cat(glob,dim=-1)
        if not self.training:
            if return_pattern==1: return ori
            if return_pattern==2: return result
            if return_pattern!=3: raise ValueError('return_pattern must be 1, 2 or 3')
            return torch.cat((ori,result),dim=-1)
        feat,aux=result
        out=[self.classifier_scfs(self.bottleneck_scfs(feat)),feat]
        if self.direct: out.extend((self.classifier(self.bottleneck(ori)),ori))
        else:
            for key,g in zip(('r','n','t'),glob): out.extend((getattr(self,'classifier_'+key)(getattr(self,'bottleneck_'+key)(g)),g))
        return tuple(out+[aux])
    def load_param(self,path):
        state=torch.load(path,map_location='cpu')
        self.load_state_dict(state.get('state_dict',state),strict=True)

def make_model(cfg,num_class,camera_num,view_num=0):
    return SCFSNet(num_class,cfg,camera_num,view_num)
