import datetime
import torch
import random
import torch.distributed as dist
import os
from .utils import print_dict
from .global_var import config
from . import nccl
import time
from .synchronize import synchronize
def init_distributed(
        init_method : str = "env://",
        seed : int = 0,
        loss_scale_factor : float = 2,
        loss_scale_steps : int = 1024,
        zero_level: int = 3,
        pipe_size: int = 1,
        num_micro_batches: int = 1
    ):
    """Initialize distributed training.
    This function will initialize the distributed training, set the random seed and global configurations.
    It must be called before any other distributed functions.

    Args:
        seed (int): The random seed.
        loss_scale_factor (float): The loss scale factor.
        loss_scale_steps (int): The loss scale steps.

    **init_distributed** reads the following environment variables: 
    
    * `WORLD_SIZE`: The total number gpus in the distributed training.
    * `RANK`: The global rank of the current gpu. From 0 to `WORLD_SIZE - 1`.
    * `MASTER_ADDR`: The address of the master node.
    * `MASTER_PORT`: The port of the master node.
    * `LOCAL_RANK`: The local rank of the current gpu.
    
    Normally, all the environments variables above are setted by the pytorch distributed launcher.

    **Note**: Do not use any functions in torch.distributed package including `torch.distributed.init_process_group` .

    **Note**: If your training script is stuck here , it means some of your distributed workers are not connected to the master node.

    """
    torch.backends.cudnn.enabled = False
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    master = os.environ["MASTER_ADDR"] + ":" + os.environ["MASTER_PORT"]
    timeout = datetime.timedelta(seconds=1800)
    rendezvous_iterator = dist.rendezvous(
        init_method, rank, world_size, timeout=timeout
    )
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)

    store = dist.PrefixStore("bmtrain", store)
    torch.cuda.set_device(local_rank)
    config["pipe_size"] = pipe_size
    config["local_rank"] = local_rank
    config["local_size"] = local_size
    config["rank"] = rank
    config["world_size"] = world_size
    config["calc_stream"] = torch.cuda.current_stream()
    config["load_stream"] = torch.cuda.Stream(priority=-1)
    config['barrier_stream'] = torch.cuda.Stream()
    config["load_event"] = torch.cuda.Event()
    config["zero_level"] = zero_level
    config["loss_scale_factor"] = loss_scale_factor if loss_scale_factor > 1 else 1 / loss_scale_factor
    config["loss_scale_steps"] = loss_scale_steps
    config["topology"] = topology(config)
    cpus_this_worker = None
    
    all_available_cpus = sorted(list(os.sched_getaffinity(0)))

    cpus_per_worker = len(all_available_cpus) // local_size
        
    if cpus_per_worker < 1:
        cpus_this_worker = all_available_cpus
        torch.set_num_threads(1)
    else:
        cpus_this_worker = all_available_cpus[local_rank * cpus_per_worker : (local_rank + 1) * cpus_per_worker]
        os.sched_setaffinity(0, cpus_this_worker)
        torch.set_num_threads( len(cpus_this_worker) )

    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    
    if rank == 0:
        unique_id : bytes = nccl.getUniqueId()
        store.set("BMTRAIN_UNIQUE_ID", unique_id.hex() )
    
    unique_id = bytes.fromhex(store.get("BMTRAIN_UNIQUE_ID").decode())
    config['comm'] = nccl.commInitRank(unique_id, world_size, rank)
    
    for i in range(world_size):
        if i == rank:
            print_dict("Initialization", {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "local_size": local_size,
                "master" : master,
                "device": torch.cuda.current_device(),
                "cpus": cpus_this_worker 
            })
        synchronize()
class topology:
    def __init__(self,config):
        self.rank = config['rank']
        pp_size = config["pipe_size"]
        world_size = config["world_size"]
        assert world_size % pp_size == 0, "The nums of GPUs must be divisible by the pipeline parallel size"

        dp_size = world_size // pp_size
        topo=torch.tensor(range(dp_size*pp_size),dtype=torch.int,device='cuda')
        topo=topo.view(pp_size,dp_size)
        self.pp_group=topo.transpose(0,1).reshape(-1,pp_size)
        self.dp_group=topo
        self.stage_id = (self.pp_group == self.rank).nonzero()[0,-1].item()
        self.stages = config['pipe_size']
        # pipe_idx is the idx of the pipeline in the group
        self.pipe_idx = (self.pp_group == self.rank).nonzero()[0, 0].item() # x axes
        self.next_rank = self.pp_group[self.pipe_idx, self.stage_id + 1].item() if self.stage_id < config['pipe_size'] - 1 else -1
        self.prev_rank = self.pp_group[self.pipe_idx, self.stage_id - 1].item() if self.stage_id > 0 else -1
        self.tails = self.pp_group[self.pipe_idx, self.stage_id:].tolist()
        self.heads = self.pp_group[self.pipe_idx, :self.stage_id + 1].tolist()
