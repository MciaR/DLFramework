import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import time

from math import ceil
from torch import optim
from data_partition import DataPartitioner
from torchvision import datasets, transforms


class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 1000)
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x


""" Partitioning MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def run(rank, size):
    """ Distributed run. """
    device = torch.device("cuda:{}".format(rank))
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = MyModel().to(device)
    # model = nn.parallel.DistributedDataParallel(model) # 这行的效果相当于81行，pytorch自动把梯度的平均给我们做了。
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.9)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    total_time = 0
    for epoch in range(10):
        epoch_loss = 0.0
        ep_start_time = time.time()
        for data, target in train_set:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            average_gradients(model) # 如果注释掉这行，就相当于每个GPU在自己跑自己的，不同步梯度，如果打开，就相当于使用了数据并行。
            optimizer.step()
        ep_end_time = time.time()
        total_time += (ep_end_time - ep_start_time)
        print('Rank {} epoch {}: {}'.format(dist.get_rank(), epoch, epoch_loss / num_batches))

    print('avg time consume per epoch: {}'.format(total_time))


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    # gloo后端是用来在CPU上进行通信的，也就是在CPU上进行分布式训练的。
    # nccl后端CUDA的Tensor操作上有较大的优化。实验发现NCCL后端比gloo后端速度更快，但显存占用更大。
    # nccl: 44.69s, 1700, 1000MB GPU Memory; 1100, 994 MB GPU Memory（注释掉81行）
    # gloo: 65.6s, 710, 570 MB GPU Memory; 706, 558 MB GPU Memory （注释掉81行）
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size) #这段代码就是将该进程加入到进程组中的核心代码
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()