import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from optimize import train_epoch,val_epoch

class YourModel(torch.nn.Module):
    def __init__(self):
        pass

class YourDataset(Dataset):
    def __init__(self,mode='train'):
        """
        mode: train/val
        """
        pass

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='pytorch ddp training template')
    # local_rank argument for torch.distributed.launch auto-assignment for multi-gpus training
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')

    # training parameters
    parser.add_argument('--start_epoch', type=int, default=-1, help='start epoch')
    parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--n_threads', type=int, default=16, help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=1027, help='Pseudo-RNG seed')
    args = parser.parse_args()

     # get current device and weather master device
    args.is_master = args.local_rank == 0
    args.device = args.local_rank
    print(args)

    # initialize PyTorch distributed using environment variables
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    # initialize random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # models
    model = YourModel()
    # send model to gpu
    model = model.to(args.local_rank)
    # initialize distributed data parallel (DDP)
    model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank)

    # train and val dataset
    train_dataset = YourDataset(mode='train')
    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                sampler=train_sampler,
                                batch_size=args.batch_size,
                                num_workers=args.n_threads)
    
    val_dataset = YourDataset(mode='val')
    val_sampler = DistributedSampler(val_dataset,shuffle=False)
    val_dataloader = DataLoader(dataset=val_dataset,
                                sampler=val_sampler,
                                batch_size=args.batch_size,
                                num_workers=args.n_threads)
    
    # Optimizer
    optimizer = []

    # training loop
    for epoch in range(args.start_epoch+1, args.n_epoch):

        # Training
        train_epoch(model,optimizer,train_dataloader,args,epoch,DDP=True)
        # validation
        val_epoch(model,val_dataloader,args,epoch,DDP=True)

        # etc

        # write log info and checkpoint
        if args.is_master:
            # etc
            pass