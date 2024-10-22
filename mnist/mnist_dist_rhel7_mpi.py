from __future__ import print_function

import os
import time
from math import ceil
from random import Random

import mpi4py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


""" Dataset partitioning helper """


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_numpy_dataset(world_size=4, world_rank=0):
    # print("Data Loading")
    dataset = np.load("datasets/train_data.npy")
    targets = np.load("datasets/train_target.npy")
    size = world_size#dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    # print("Partition Sizes {}".format(partition_sizes))
    partition_data = DataPartitioner(dataset, partition_sizes)
    partition_data = partition_data.use(world_rank)
    train_set_data = torch.utils.data.DataLoader(partition_data,
                                                 batch_size=bsz,
                                                 shuffle=False)
    partition_target = DataPartitioner(targets, partition_sizes)
    partition_target = partition_target.use(world_rank)
    train_set_target = torch.utils.data.DataLoader(partition_target,
                                                   batch_size=bsz,
                                                   shuffle=False)
    return train_set_data, train_set_target, bsz


def partition_numpy_dataset_test(world_rank=0, world_size=4):
    # print("Data Loading")
    dataset = np.load("datasets/test_data.npy")
    targets = np.load("datasets/test_target.npy")
    # print("Data Size For Test {} {}".format(dataset.shape, targets.shape))

    bsz = int(16 / float(world_size))
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    # print("Partition Sizes {}".format(partition_sizes))
    partition_data = DataPartitioner(dataset, partition_sizes)
    partition_data = partition_data.use(world_rank)
    train_set_data = torch.utils.data.DataLoader(partition_data,
                                                 batch_size=bsz,
                                                 shuffle=False)
    partition_target = DataPartitioner(targets, partition_sizes)
    partition_target = partition_target.use(world_rank)
    train_set_target = torch.utils.data.DataLoader(partition_target,
                                                   batch_size=bsz,
                                                   shuffle=False)

    return train_set_data, train_set_target, bsz


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_accuracy(local_accuracy):
    size = float(dist.get_world_size())
    dist.all_reduce(local_accuracy, op=dist.ReduceOp.SUM)
    global_accuracy = local_accuracy / size
    return global_accuracy


def average_gradients_mpi(model, comm=None, world_size=4):
    size = float(world_size)
    for param in model.parameters():
        param_numpy = param.grad.data.numpy()
        param_output = np.empty(param_numpy.shape, dtype=param_numpy.dtype)
        param_torch = torch.from_numpy(param_numpy)
        #print(type(param.grad.data), type(param_numpy), type(param_torch), param_numpy.dtype, param_output.dtype)
        comm.Allreduce(param_numpy, param_output, op=MPI.SUM)
        param.grad.data = torch.from_numpy(param_output)
        #dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_accuracy_mpi(local_accuracy, comm=None, world_size=4):
    size = float(world_size)
    #dist.all_reduce(local_accuracy, op=dist.ReduceOp.SUM)
    param_numpy = local_accuracy.numpy()
    param_output = np.empty(param_numpy.shape, dtype=param_numpy.dtype)
    comm.Allreduce(param_numpy, param_output, op=MPI.SUM)
    local_accuracy = torch.from_numpy(param_output)
    global_accuracy = local_accuracy / size
    return global_accuracy


""" Distributed Synchronous SGD Example """


def run(rank, size, do_log=False, comm=None):
    if (rank == 0):
        print("Run Fn")

    torch.manual_seed(1234)
    train_set_data, train_set_target, bsz = partition_numpy_dataset(world_size=size, world_rank=rank)
    if (do_log):
        print("Data Points Per Rank {} of Size {}".format(len(train_set_data.dataset), size))
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set_data.dataset) / float(bsz))
    if (rank == 0):
        print("Started Training")
    total_data = len(train_set_data)
    epochs = 20
    total_steps = epochs * total_data
    local_time_communication = 0
    local_total_time_communication = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        count = 0
        for data, target in zip(train_set_data, train_set_target):
            data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2])) / 128.0
            # print(
            #     "Data Size {}({},{}) of Rank {} : target {}, {}".format(data.shape, (data[0].numpy().dtype), type(data),
            #                                                             rank, target, len(target)))
            # print(data[0], target[0])
            count = count + 1
            result = '{0:.4g}'.format((count / float(total_steps)) * 100.0)
            if (rank == 0):
                print("Progress {}% \r".format(result), end='\r')
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            # print(epoch_loss)
            loss.backward()
            if (rank == 0):
                local_time_communication = time.time()
            average_gradients_mpi(model, comm=comm, world_size=size)
            if (rank == 0):
                local_time_communication = time.time() - local_time_communication
                local_total_time_communication = local_total_time_communication + local_time_communication
            optimizer.step()
        if (rank == 0):
            print('Rank ', rank, ', epoch ',
                  epoch, ': ', epoch_loss / num_batches)
    return model, local_total_time_communication


def test(rank, size, model, device, do_log=False, comm=None):
    test_set_data, test_set_target, bsz = partition_numpy_dataset_test(world_rank=rank, world_size=size)
    model.eval()
    test_loss = 0
    correct = 0
    # print(test_set_data)
    total_samples = 0
    val1 = 0
    val2 = 0
    count = 0
    with torch.no_grad():
        for data, target in zip(test_set_data, test_set_target):
            # total_samples = total_samples + 1
            count = count + 1
            val1 = len(data)
            val2 = len(test_set_data)
            total_samples = (val1 * val2)
            data, target = data.to(device), target.to(device)
            data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2])) / 128.0
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if (rank == 0 and do_log):
                print(rank, count, len(data), len(test_set_data), data.shape, output.shape, correct, total_samples)

    test_loss /= (total_samples)
    local_accuracy = 100.0 * correct / total_samples
    global_accuracy = average_accuracy_mpi(torch.tensor(local_accuracy), comm=comm, world_size=size)
    if (rank == 0):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, total_samples,
            global_accuracy.numpy()))


def save_log(file_path=None, stat=""):
    fp = open(file_path, mode="a+")
    fp.write(stat + "\n")
    fp.close()


def init_processes(rank, size, fn, backend='tcp', do_log=False, comm=None):
    """ Initialize the distributed environment. """
    #dist.init_process_group(backend, rank=rank, world_size=size)
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    # model1 = Net()
    # test(rank, model1, device)
    total_communication_time = 0
    local_training_time = 0
    local_testing_time = 0
    if (rank == 0):
        local_training_time = time.time()
    model, total_communication_time = fn(rank, size, comm=comm)
    if (rank == 0):
        local_training_time = time.time() - local_training_time
    if (rank == 0):
        local_testing_time = time.time()
    test(rank, size, model, device, do_log=do_log, comm=comm)
    if (rank == 0):
        local_testing_time = time.time() - local_testing_time
        print("Total Training Time : {}".format(local_training_time))
        print("Total Testing Time : {}".format(local_testing_time))
        save_log("stats.csv",
                 stat="{},{},{},{}".format(size, local_training_time, total_communication_time, local_testing_time))


if __name__ == "__main__":
    do_log = False
    MPI.Init()
    comm = MPI.COMM_WORLD
    parent = comm.Get_parent()
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    # print(world_rank, world_size)
    init_processes(world_rank, world_size, run, backend='mpi', do_log=do_log, comm=comm)
    MPI.Finalize()
