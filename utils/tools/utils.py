import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    """Use distributed training when launched with torchrun (or SLURM), otherwise a single GPU."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.gpu = 0
        return

    args.distributed = True
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
