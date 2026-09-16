import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


class Nccl:
    def __init__(self, options):
        self.options = options
        local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device("cuda", local_rank)
        self.rank = local_rank + options.node_rank * options.sender_ranks

    def __enter__(self):
        torch.cuda.set_device(self.device)
        world_size = self.options.sender_ranks + 8
        timeout = timedelta(seconds=180)
        store = dist.TCPStore(
            self.options.master_addr, 29500, world_size,
            is_master=self.rank == 0, timeout=timeout,
        )
        dist.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=world_size,
            device_id=self.device, timeout=timeout,
        )
        return self

    def __exit__(self, *exception):
        dist.destroy_process_group()


class Broadcast:
    def __init__(self, nccl):
        self.tensor = torch.full(
            (512 * 1024**2,), nccl.rank == 0,
            dtype=torch.bfloat16, device=nccl.device,
        )
        self.payload_gb = self.tensor.numel() * self.tensor.element_size() / 1e9

    def run(self):
        dist.broadcast(self.tensor, src=0)

    def validate(self):
        if not bool(torch.all(self.tensor == 1)):
            raise RuntimeError(f"rank {dist.get_rank()}: incorrect payload")


class Profile:
    def __init__(self, nccl):
        self.nccl = nccl
        self.iterations = 100

    def measure(self, broadcast):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(self.iterations):
            broadcast.run()
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - started, device=self.nccl.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return broadcast.payload_gb * self.iterations / elapsed.item()

    def run(self, broadcast, seconds):
        for _ in range(5):
            broadcast.run()
        stop = torch.zeros((), dtype=torch.int32, device=self.nccl.device)
        started = time.monotonic()
        while not stop.item():
            throughput_gb_per_second = self.measure(broadcast)
            if self.nccl.rank == 0:
                print(f"{throughput_gb_per_second:.2f} GB/s", flush=True)
                stop.fill_(time.monotonic() - started >= seconds)
            dist.broadcast(stop, src=0)
        broadcast.validate()
        dist.barrier()
        if self.nccl.rank == 0:
            print("PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--sender-ranks", type=int, choices=(1, 8), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--seconds", type=float, default=300)
    options = parser.parse_args()
    with Nccl(options) as nccl:
        Profile(nccl).run(Broadcast(nccl), options.seconds)


if __name__ == "__main__":
    main()
