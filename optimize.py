import torch
from tqdm import tqdm
import torch.distributed as dist



def train_epoch(net,optimizer,train_loader,args,epoch,DDP='True'):
    """
    training epoch for net
    """
    is_master = (not DDP) or args.is_master  #if not a ddp train or is master in ddp
    net.train()
    if DDP:
        dist.barrier()
    datalist = enumerate(train_loader)
    if is_master:
        datalist = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in datalist:
        optimizer.zero_grad()

        ###net forward and backward

        #etc

        optimizer.step()

        if is_master: ### train log should only run on master
            pass

def val_epoch(net,val_loader,args,epoch,DDP='True'):
    """
    valuation epoch for net
    """
    is_master = (not DDP) or args.is_master  #if not a ddp train or is master in ddp
    net.eval()
    if DDP:
        dist.barrier()
    
    with torch.no_grad():
        datalist = enumerate(val_loader)
        if is_master:
            datalist = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, mini_batch in datalist:
            
            ###net forward for val

            #etc
            pass

            if is_master: ### val log should only run on master
                pass
