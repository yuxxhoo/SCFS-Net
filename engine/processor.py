"""Training and retrieval evaluation for SCFS-Net."""
import json
import logging
from pathlib import Path
import torch
from torch.cuda import amp
from utils.metrics import R1_mAP,R1_mAP_eval

@torch.no_grad()
def evaluate(cfg,model,loader,num_query,device,pattern=3):
    model.eval()
    evaluator=(R1_mAP(num_query,feat_norm=cfg.TEST.FEAT_NORM) if cfg.DATASETS.NAMES=='MSVR310' else R1_mAP_eval(num_query,feat_norm=cfg.TEST.FEAT_NORM=='yes',reranking=cfg.TEST.RE_RANKING=='yes'))
    evaluator.reset()
    for img,pid,cam,cam_batch,view,paths in loader:
        feat=model({k:v.to(device) for k,v in img.items()},cam_label=cam_batch.to(device),view_label=view.to(device),return_pattern=pattern)
        evaluator.update((feat.cpu(),pid,cam,view,paths) if cfg.DATASETS.NAMES=='MSVR310' else (feat.cpu(),pid,cam,paths))
    cmc,map_value,*_=evaluator.compute()
    result={'mAP':float(map_value)*100}
    for rank in (1,5,10):
        if rank<=len(cmc): result[f'Rank-{rank}']=float(cmc[rank-1])*100
    return result

def train(cfg,model,train_loader,val_loader,num_query,optimizer,scheduler,loss_fn,device):
    output=Path(cfg.OUTPUT_DIR); output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('SCFSNet')
    scaler=amp.GradScaler(enabled=device.type=='cuda'); best=-1
    for epoch in range(1,cfg.SOLVER.MAX_EPOCHS+1):
        scheduler.step(epoch); model.train(); total=0
        for step,(img,pid,cam,view,_) in enumerate(train_loader,1):
            optimizer.zero_grad()
            target=pid.to(device)
            with amp.autocast(enabled=device.type=='cuda'):
                values=model({k:v.to(device) for k,v in img.items()},label=target,cam_label=cam.to(device),view_label=view.to(device))
                loss=sum(loss_fn(score=values[i],feat=values[i+1],target=target,target_cam=cam.to(device)) for i in range(0,len(values)-1,2))+values[-1]
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total+=float(loss.detach())
            if step%cfg.SOLVER.LOG_PERIOD==0: logger.info('Epoch %d step %d/%d loss %.4f',epoch,step,len(train_loader),total/step)
        if epoch%cfg.SOLVER.EVAL_PERIOD==0:
            record={'epoch':epoch,'loss':total/max(step,1)}
            for pattern in (1,2,3):
                result=evaluate(cfg,model,val_loader,num_query,device,pattern)
                record[f'pattern_{pattern}']=result
                logger.info('Epoch %d pattern %d: %s',epoch,pattern,result)
            with (output/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
            if record['pattern_3']['mAP']>best:
                best=record['pattern_3']['mAP']; torch.save(model.state_dict(),output/'SCFSNet_best.pth')
        if epoch%cfg.SOLVER.CHECKPOINT_PERIOD==0: torch.save(model.state_dict(),output/f'SCFSNet_epoch_{epoch}.pth')
