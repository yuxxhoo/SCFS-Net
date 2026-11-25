"""Evaluate a user-provided SCFS-Net checkpoint."""
import json
import torch
from tools.common import options,seed_everything
from data import make_dataloader
from modeling import make_model
from engine.processor import evaluate

def main():
    cfg=options('SCFS-Net evaluation; set MODEL.INIT_WEIGHTS False and TEST.WEIGHT checkpoint')
    seed_everything(cfg.SOLVER.SEED)
    if not cfg.TEST.WEIGHT: raise ValueError('TEST.WEIGHT must specify a trained SCFS-Net checkpoint')
    _,_,val,nquery,nclass,ncam,nview=make_dataloader(cfg)
    device=torch.device(cfg.MODEL.DEVICE)
    model=make_model(cfg,nclass,ncam,nview).to(device)
    model.load_param(cfg.TEST.WEIGHT)
    print(json.dumps(evaluate(cfg,model,val,nquery,device),indent=2))
if __name__=='__main__': main()
