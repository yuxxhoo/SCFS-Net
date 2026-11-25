"""Complete spectral partition with amplitude and phase statistics."""
import torch
from torch import nn
import torch.nn.functional as F

class CAFD(nn.Module):
    """Complete Asymmetric Identity–Modality Discrepancy Decomposition (CAFD)."""
    def __init__(self, dim, stalk_dim, cfg):
        super().__init__()
        s = cfg.MODEL.SCFS
        self.bands = int(s.NUM_BANDS)
        self.stalk_dim = stalk_dim
        self.mask_logits = nn.Parameter(torch.zeros(3, self.bands))
        # Ordered initialization: structure-low, detail-high, modality-mid.
        with torch.no_grad():
            self.mask_logits[0] = torch.linspace(2, -2, self.bands)
            self.mask_logits[1] = torch.linspace(-2, 2, self.bands)
            self.mask_logits[2] = -torch.linspace(-1, 1, self.bands).abs()
        self.branch_encoder = nn.ModuleList([nn.Sequential(nn.LayerNorm(3*dim), nn.Linear(3*dim, stalk_dim), nn.GELU(), nn.Linear(stalk_dim, stalk_dim)) for _ in range(3)])
        self.last_partition = None
        self.last_completeness = None
        self.last_decorrelation = None

    def radial_basis(self, h, w, device):
        fy=torch.fft.fftfreq(h,device=device).abs()[:,None]; fx=torch.fft.rfftfreq(w,device=device).abs()[None,:]
        r=torch.sqrt(fy.square()+fx.square()); r/=r.max().clamp_min(1e-6)
        centers=torch.linspace(0,1,self.bands,device=device); width=1/max(self.bands-1,1)
        basis=torch.stack([torch.exp(-0.5*((r-c)/width).square()) for c in centers]); return basis/basis.sum(0,keepdim=True).clamp_min(1e-6)

    def forward(self, maps):
        spectrum=torch.fft.rfft2(maps.float(),norm="ortho"); amp=spectrum.abs(); unit=spectrum/amp.clamp_min(1e-6)
        basis=self.radial_basis(maps.shape[-2],maps.shape[-1],maps.device)
        branch_band=self.mask_logits.softmax(0)  # exactly complete for every basis band
        masks=torch.einsum("cb,bhw->chw",branch_band,basis)
        count=masks.sum((-2,-1)).clamp_min(1e-6)
        states=[]
        for branch in range(3):
            a=torch.einsum("bmsdhw,hw->bmsd",amp,masks[branch])/count[branch]
            c=torch.einsum("bmsdhw,hw->bmsd",unit.real,masks[branch])/count[branch]
            si=torch.einsum("bmsdhw,hw->bmsd",unit.imag,masks[branch])/count[branch]
            states.append(self.branch_encoder[branch](torch.cat((torch.log1p(a),c,si),-1)))
        states=torch.stack(states,1) # B x 3 branches x 3 modalities x 3 scales x d
        identity=(states[:,0]+states[:,1]).reshape(states.shape[0],-1,self.stalk_dim)
        modality=states[:,2].reshape(states.shape[0],-1,self.stalk_dim)
        identity=identity-identity.mean(1,keepdim=True); modality=modality-modality.mean(1,keepdim=True)
        cross=torch.einsum("bnd,bne->bde",identity,modality)/identity.shape[1]
        decorrelation=cross.square().mean()
        completeness=(masks.sum(0)-1).abs().mean()
        self.last_partition=branch_band.detach(); self.last_completeness=completeness.detach(); self.last_decorrelation=decorrelation.detach()
        return states, decorrelation, completeness
