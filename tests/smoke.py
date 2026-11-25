"""CPU integration and gradient smoke test; no dataset or weights needed."""
from pathlib import Path
import torch
from config import get_cfg_defaults
from modeling import make_model
from layers.make_loss import make_loss

def main():
    torch.set_num_threads(4); torch.manual_seed(7)
    root=Path(__file__).resolve().parents[1]
    for name in ('RGBNT201','RGBNT100','MSVR310'):
        cfg=get_cfg_defaults();cfg.merge_from_file(str(root/'configs'/f'{name}.yml'))
        assert all((cfg.MODEL.SCFS.USE_FSCA,cfg.MODEL.SCFS.USE_CAFD_DISENTANGLEMENT,cfg.MODEL.SCFS.USE_MHSR))
    cfg.MODEL.INIT_WEIGHTS=False; cfg.MODEL.DEVICE='cpu'
    model=make_model(cfg,2,2)
    x={k:torch.randn(4,3,*cfg.INPUT.SIZE_TRAIN) for k in ('RGB','NI','TI')}
    target=torch.tensor([0,0,1,1]);cam=torch.zeros(4,dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for pattern,width in ((1,1536),(2,1536),(3,3072)):
            y=model(x,cam_label=cam,return_pattern=pattern)
            assert y.shape==(4,width) and torch.isfinite(y).all()
    model.train(); result=model(x,cam_label=cam)
    loss_fn,_=make_loss(cfg,2)
    loss=sum(loss_fn(score=result[i],feat=result[i+1],target=target,target_cam=cam) for i in range(0,len(result)-1,2))+result[-1]
    assert torch.isfinite(loss);loss.backward()
    for module in (model.collaboration.fusion.fsca,model.collaboration.fusion.cafd,model.collaboration.fusion.mhsr):
        gradients=[v.grad for v in module.parameters() if v.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(v.grad is not None for v in model.backbone.parameters())
    print('PASS: three configs, complete model evaluation/training, finite gradients in FSCA/CAFD/MHSR and backbone.')
if __name__=='__main__':main()
