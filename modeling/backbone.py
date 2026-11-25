"""Shared CLIP visual backbone. See THIRD_PARTY_NOTICES.md."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .clip_visual import VisionTransformer

class Backbone(nn.Module):
    def __init__(self, cfg, camera_num):
        super().__init__()
        if cfg.MODEL.FROZEN:
            raise ValueError("This release supports the verified FROZEN=False protocol.")
        h,w=cfg.INPUT.SIZE_TRAIN
        stride=cfg.MODEL.STRIDE_SIZE[0]
        gh,gw=(h-16)//stride+1,(w-16)//stride+1
        self.base=VisionTransformer(gh,gw,16,stride,768,12,12,512,cfg)
        self.sie=float(cfg.MODEL.SIE_COE)
        self.cv_embed=None
        if cfg.MODEL.SIE_CAMERA:
            self.cv_embed=nn.Parameter(torch.zeros(camera_num,1,768))
            nn.init.trunc_normal_(self.cv_embed,std=.02)
        if cfg.MODEL.INIT_WEIGHTS:
            path=Path(cfg.MODEL.CLIP_PATH)
            if not path.is_file():
                raise FileNotFoundError(f"Missing external CLIP initialization: {path}. Run python -m tools.download_backbone.")
            try:
                state=torch.jit.load(str(path),map_location='cpu').state_dict()
            except RuntimeError:
                state=torch.load(path,map_location='cpu')
            state={k[7:]:v.float() for k,v in state.items() if k.startswith('visual.')}
            pos=state['positional_embedding']
            side=int((pos.shape[0]-1)**.5)
            grid=pos[1:].reshape(side,side,-1).permute(2,0,1).unsqueeze(0)
            grid=F.interpolate(grid,size=(gh,gw),mode='bilinear',align_corners=False)
            state['positional_embedding']=torch.cat((pos[:1],grid.squeeze(0).permute(1,2,0).reshape(gh*gw,-1)))
            self.base.load_state_dict(state,strict=True)
    def forward(self,x,cam_label=None,view_label=None):
        cv=None
        if self.cv_embed is not None:
            if cam_label is None: raise ValueError("Camera labels are required when SIE_CAMERA=True")
            cv=self.sie*self.cv_embed[cam_label]
        tokens=self.base(x,cv)
        return tokens[:,1:],tokens[:,0]
