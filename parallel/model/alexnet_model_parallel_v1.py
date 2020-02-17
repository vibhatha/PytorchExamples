import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timeit


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1_relu = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.fc2_relu = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x


class ModelParallelAlexNet(AlexNet):

    def __init__(self, num_classes, devices_layer_mapping=['cuda:0', 'cuda:1']):
        super(ModelParallelAlexNet, self).__init__(num_classes=num_classes)
        self.features = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
            self.conv2,
            self.relu2,
            self.maxpool2,
            self.conv3,
            self.relu3,
            self.conv4,
            self.relu4,
            self.conv5,
            self.relu5,
            self.maxpool3,

        ).to(devices_layer_mapping[0])

        self.pool = nn.Sequential(
            self.avgpool
        ).to(devices_layer_mapping[0])

        self.classifier = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
            self.dropout2,
            self.fc2,
            self.fc2_relu,
            self.fc3
        ).to(devices_layer_mapping[1])

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1).to('cuda:1')
        x = self.classifier(x)
        return x


num_classes = 1000
num_batches = 1
batch_size = 120
image_w = 128
image_h = 128
num_repeat = 3

cuda_available = torch.cuda.is_available()

print("===================================================")
print("Cuda Available : {}".format(cuda_available))
print("===================================================")


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
            .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        if cuda_available:
            outputs = model(inputs.to('cuda:0'))
        else:
            outputs = model(inputs)
        # print("Output-device {}".format(outputs.device))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


stmt = "train(model)"
setup = None

if cuda_available:
    setup = "model = ModelParallelAlexNet(num_classes=num_classes).to('cuda:0')"
else:
    setup = "model = ModelParallelAlexNet(num_classes=num_classes)"

rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

print("Model Parallel Training Time:", rn_mean)
