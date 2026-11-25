"""Train SCFS-Net on RGBNT201, RGBNT100 or MSVR310."""
import logging
from pathlib import Path
import torch
from tools.common import options,seed_everything
from data import make_dataloader
from modeling import make_model
from layers.make_loss import make_loss
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from engine.processor import train

def main():
    cfg=options('SCFS-Net training'); seed_everything(cfg.SOLVER.SEED)
    output=Path(cfg.OUTPUT_DIR); output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(output/'train.log')])
    (output/'config.yml').write_text(cfg.dump())
    loader,_,val,nquery,nclass,ncam,nview=make_dataloader(cfg)
    device=torch.device(cfg.MODEL.DEVICE)
    model=make_model(cfg,nclass,ncam,nview).to(device)
    loss,center=make_loss(cfg,nclass)
    optimizer,_=make_optimizer(cfg,model,center)
    train(cfg,model,loader,val,nquery,optimizer,create_scheduler(cfg,optimizer),loss,device)
if __name__=='__main__': main()
