import torch
from tqdm import tqdm
import torch.distributed as dist



def train_epoch(net,optimizer,train_loader,args,epoch,DDP='True'):
    """
    training epoch for net
    """
    net.train()
    if DDP:
        dist.barrier()
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        optimizer.zero_grad()

        ###net forward and backward

        #etc

        optimizer.step()

        if (not DDP) or args.is_master: ### train log should only run on rank 0
            pass

def val_epoch(net,val_loader,args,epoch,DDP='True'):
    """
    valuation epoch for net
    """
    net.eval()
    if DDP:
        dist.barrier()
    
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, mini_batch in pbar:
            
            ###net forward for val

            #etc
            pass

        if (not DDP) or args.is_master: ### val log should only run on rank 0
            pass
