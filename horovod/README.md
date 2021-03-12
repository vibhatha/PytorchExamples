# Running Horovod Simple Example 

## Code

```python
from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.nn as nn
import torch.optim as optim

import horovod.torch as hvd

NUM_SAMPLES = 5
INPUT_FEATURES = 5
OUT_FEATURES = 5
HIDDEN_FEATURES = 2
NUM_OF_BATCHES = 5


class SampleDataSet(Dataset):

    def __init__(self, rank):
        self.rank = rank
        self.data = [self._get_datum() for _ in range(NUM_OF_BATCHES)]

    def _get_datum(self):
        return np.ones(NUM_SAMPLES * INPUT_FEATURES, dtype=np.float32).reshape(NUM_SAMPLES, INPUT_FEATURES) * self.rank

    def __getitem__(self, item):
        datum = self.data[item]
        return datum

    def __len__(self):
        return len(self.data)


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(HIDDEN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train(rank, world_size, data, model, loss_fn, optimizer):
    print(f"Running basic DDP example on rank {rank} : {data}.")
    # create model and move it to GPU with id rank
    optimizer.zero_grad()
    outputs = model(data)
    labels = torch.randn(data.shape)
    loss_fn(outputs, labels).backward()
    optimizer.step()


hvd.init()

rank = hvd.rank()
world_size = hvd.size()

dataset = SampleDataSet(rank=rank)

data_loader = DataLoader(dataset=dataset)

# Horovod: broadcast parameters & optimizer state.

model = ToyModel()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum)

for batch_id, data in enumerate(data_loader):
    print(batch_id, type(data), data.shape)

    train(rank, world_size, data, model, loss_fn, optimizer)

```

## RUN with MPI

```bash
horovodrun --mpi -n 4 python horovod/horovod_sample.py
```