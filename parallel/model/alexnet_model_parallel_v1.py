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
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc2_relu(x)

        x = self.fc3(x)

        return x


class ModelParallelAlexNetV1(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1']):
        super(ModelParallelAlexNetV1, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
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
            self.avgpool
        ).to(devices[0])
        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq2 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[1])

        self.fc3.to(devices[1])

    def forward(self, x):
        x = self.seq2(torch.flatten(self.seq1(x), 1).to('cuda:1'))
        return self.fc3(x)


class ModelParallelAlexNetV2(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1',
                                                  'cuda:2']):
        super(ModelParallelAlexNetV2, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
            self.conv2,
            self.relu2,
            self.maxpool2,
        ).to(devices[0])

        self.seq2 = nn.Sequential(
            self.conv3,
            self.relu3,
            self.conv4,
            self.relu4,
            self.conv5,
            self.relu5,
            self.maxpool3,
            self.avgpool
        ).to(devices[1])

        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq3 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[2])

        self.fc3.to(devices[2])

    def forward(self, x):
        x = self.seq1(x).to('cuda:1')
        x = self.seq3(torch.flatten(self.seq2(x), 1).to('cuda:2'))
        return self.fc3(x)


class ModelParallelAlexNetV3(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1',
                                                  'cuda:2', 'cuda:3']):
        super(ModelParallelAlexNetV3, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
            self.conv2,
            self.relu2,
            self.maxpool2,
        ).to(devices[0])

        self.seq2 = nn.Sequential(
            self.conv3,
            self.relu3,
            self.conv4,
            self.relu4,
        ).to(devices[1])

        self.seq3 = nn.Sequential(
            self.conv5,
            self.relu5,
            self.maxpool3,
            self.avgpool
        ).to(devices[2])

        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq4 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[3])

        self.fc3.to(devices[3])

    def forward(self, x):
        x = self.seq1(x).to('cuda:1')
        x = self.seq2(x).to('cuda:2')
        x = self.seq4(torch.flatten(self.seq3(x), 1).to('cuda:3'))
        return self.fc3(x)


class ModelParallelAlexNetV4(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1',
                                                  'cuda:2', 'cuda:3',
                                                  'cuda:4']):
        super(ModelParallelAlexNetV4, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
        ).to(devices[0])

        self.seq2 = nn.Sequential(
            self.conv2,
            self.relu2,
            self.maxpool2,
        ).to(devices[1])

        self.seq3 = nn.Sequential(
            self.conv3,
            self.relu3,
            self.conv4,
            self.relu4,
        ).to(devices[2])

        self.seq4 = nn.Sequential(
            self.conv5,
            self.relu5,
            self.maxpool3,
            self.avgpool
        ).to(devices[3])

        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq5 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[4])

        self.fc3.to(devices[4])

    def forward(self, x):
        x = self.seq1(x).to('cuda:1')
        x = self.seq2(x).to('cuda:2')
        x = self.seq3(x).to('cuda:3')
        x = self.seq5(torch.flatten(self.seq4(x), 1).to('cuda:4'))
        return self.fc3(x)


class ModelParallelAlexNetV5(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1',
                                                  'cuda:2', 'cuda:3',
                                                  'cuda:4', 'cuda:5']):
        super(ModelParallelAlexNetV5, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
        ).to(devices[0])

        self.seq2 = nn.Sequential(
            self.conv2,
            self.relu2,
            self.maxpool2,
        ).to(devices[1])

        self.seq3 = nn.Sequential(
            self.conv3,
            self.relu3,
        ).to(devices[2])

        self.seq4 = nn.Sequential(
            self.conv4,
            self.relu4,
        ).to(devices[3])

        self.seq5 = nn.Sequential(
            self.conv5,
            self.relu5,
            self.maxpool3,
            self.avgpool
        ).to(devices[4])

        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq6 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[5])

        self.fc3.to(devices[5])

    def forward(self, x):
        x = self.seq1(x).to('cuda:1')
        x = self.seq2(x).to('cuda:2')
        x = self.seq3(x).to('cuda:3')
        x = self.seq4(x).to('cuda:4')
        x = self.seq6(torch.flatten(self.seq5(x), 1).to('cuda:5'))
        return self.fc3(x)


class ModelParallelAlexNetV6(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1',
                                                  'cuda:2', 'cuda:3',
                                                  'cuda:4', 'cuda:5',
                                                  'cuda:6']):
        super(ModelParallelAlexNetV6, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
        ).to(devices[0])

        self.seq2 = nn.Sequential(
            self.conv2,
            self.relu2,
            self.maxpool2,
        ).to(devices[1])

        self.seq3 = nn.Sequential(
            self.conv3,
            self.relu3,
        ).to(devices[2])

        self.seq4 = nn.Sequential(
            self.conv4,
            self.relu4,
        ).to(devices[3])

        self.seq5 = nn.Sequential(
            self.conv5,
            self.relu5,
            self.maxpool3,
            self.avgpool
        ).to(devices[4])

        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq6 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
        ).to(devices[5])

        self.seq7 = nn.Sequential(
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[6])

        self.fc3.to(devices[6])

    def forward(self, x):
        x = self.seq1(x).to('cuda:1')
        x = self.seq2(x).to('cuda:2')
        x = self.seq3(x).to('cuda:3')
        x = self.seq4(x).to('cuda:4')
        x = self.seq6(torch.flatten(self.seq5(x), 1).to('cuda:5')).to('cuda:6')
        x = self.seq7(x)
        return self.fc3(x)


class ModelParallelAlexNetV7(AlexNet):

    def __init__(self, num_classes=1000, devices=['cuda:0', 'cuda:1',
                                                  'cuda:2', 'cuda:3',
                                                  'cuda:4', 'cuda:5',
                                                  'cuda:6', 'cuda:7']):
        super(ModelParallelAlexNetV7, self).__init__(num_classes=num_classes)
        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu1,
            self.maxpool1,
        ).to(devices[0])

        self.seq2 = nn.Sequential(
            self.conv2,
            self.relu2,
            self.maxpool2,
        ).to(devices[1])

        self.seq3 = nn.Sequential(
            self.conv3,
            self.relu3,
        ).to(devices[2])

        self.seq4 = nn.Sequential(
            self.conv4,
            self.relu4,
        ).to(devices[3])

        self.seq5 = nn.Sequential(
            self.conv5,
            self.relu5,
            self.maxpool3,
            self.avgpool
        ).to(devices[4])

        #
        # self.pool = nn.Sequential(
        #     self.avgpool
        # ).to(devices_layer_mapping[0])

        self.seq6 = nn.Sequential(
            self.dropout1,
            self.fc1,
            self.fc1_relu,
        ).to(devices[5])

        self.seq7 = nn.Sequential(
            self.dropout2,
            self.fc2,
            self.fc2_relu,
        ).to(devices[6])

        self.fc3.to(devices[7])

    def forward(self, x):
        x = self.seq1(x).to('cuda:1')
        x = self.seq2(x).to('cuda:2')
        x = self.seq3(x).to('cuda:3')
        x = self.seq4(x).to('cuda:4')
        x = self.seq6(torch.flatten(self.seq5(x), 1).to('cuda:5')).to('cuda:6')
        x = self.seq7(x).to('cuda:7')
        return self.fc3(x)


num_classes = 1000
num_batches = 1
batch_size = 120
image_w = 128
image_h = 128
num_repeat = 10

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
        outputs = model(inputs.to('cuda:0'))

        # print("Output-device {}".format(outputs.device))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


stmt = "train(model)"

for version in range(7):
    setup = "model = ModelParallelAlexNetV{}(num_classes=num_classes)".format(
        version+1)

    rn_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

    print("Devices {}: Model Parallel V{} Training Time: {}".format(version,
          str(version + 1),
          rn_mean) )
