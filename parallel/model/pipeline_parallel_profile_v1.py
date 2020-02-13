import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime, date, time


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise Exception('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise Exception("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise Exception("replace_stride_with_dilation should be None "
                            "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


#################### DL CODE ################

num_classes = 1000


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))


class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        mb_device_0_start_time = 0
        mb_device_0_start_datetime = datetime.now()
        mb_device_0_end_time = 0
        mb_device_0_end_datetime = datetime.now()
        mb_device_1_start_time = 0
        mb_device_1_start_datetime = datetime.now()
        mb_device_1_end_time = 0
        mb_device_1_end_datetime = datetime.now()
        mb_fc_start_time = 0
        mb_fc_start_datetime = datetime.now()
        mb_fc_end_time = 0
        mb_fc_end_datetime = datetime.now()
        c0_c1_cp_start_time = 0
        co_c1_cp_start_datetime = datetime.now()
        c0_c1_cp_end_time = 0
        c0_c1_cp_end_datetime = datetime.now()
        seq1_time = 0
        seq1_datetime = datetime.now()
        c0_c1_copy_time = 0
        co_c1_copy_datetime = datetime.now()
        t1 = time.time()
        splits = iter(x.split(self.split_size, dim=0))
        t2 = time.time()
        split_time = t2 - t1
        s_next = next(splits)
        mb_device_0_start_time = time.time()        
        mb_device_0_start_datetime = datetime.now()
        s_prev = self.seq1(s_next)
        mb_device_0_end_time = time.time()        
        mb_device_0_end_datetime = datetime.now()
        seq1_time = mb_device_0_end_time - mb_device_0_start_time        
        c0_c1_cp_start_time = time.time()
        c0_c1_cp_start_datetime = datetime.now()
        s_prev = s_prev.to('cuda:1')
        c0_c1_cp_end_time = time.time()        
        c0_c1_cp_end_datetime = datetime.now()
        c0_c1_copy_time = c0_c1_cp_end_time - c0_c1_cp_start_time
        ret = []
        seq2_time = 0
        seq2_datetime = datetime.now()
        seq_fc_time = 0
        seq_fc_datetime = datetime.now()
        split_id = 1
        print(split_id,seq1_time, c0_c1_copy_time, seq2_time, seq_fc_time, mb_device_0_start_time, mb_device_0_end_time, c0_c1_cp_start_time, c0_c1_cp_end_time, mb_device_1_start_time, mb_device_1_end_time, mb_fc_start_time, mb_fc_end_time, mb_device_0_start_datetime, mb_device_0_end_datetime, c0_c1_cp_start_datetime, c0_c1_cp_end_datetime, mb_device_1_start_datetime, mb_device_1_end_datetime, mb_fc_start_datetime, mb_fc_end_datetime)

        for s_next in splits:
            # A. s_prev runs on cuda:1
            mb_device_1_start_time = time.time()
            mb_device_1_start_datetime = datetime.now()
            s_prev = self.seq2(s_prev)
            mb_device_1_end_time = time.time()
            mb_device_1_end_datetime = datetime.now()
            seq2_time = mb_device_1_end_time - mb_device_1_start_time        
            mb_fc_start_time = time.time()
            mb_fc_start_datetime = datetime.now()
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
            mb_fc_end_time = time.time()
            mb_fc_end_datetime = datetime.now()
            seq_fc_time = mb_fc_end_time - mb_fc_start_time

            # B. s_next runs on cuda:0, which can run concurrently with A
            mb_device_0_start_time = time.time()
            mb_device_0_start_datetime = datetime.now()
            s_prev = self.seq1(s_next)
            mb_device_0_end_time = time.time()
            mb_device_0_end_datetime = datetime.now()
            seq1_time = mb_device_0_end_time - mb_device_0_start_time
            c0_c1_cp_start_time = time.time()
            c0_c1_cp_start_datetime = datetime.now()
            s_prev = s_prev.to('cuda:1')
            c0_c1_cp_end_time = time.time()
            c0_c1_cp_end_datetime = datetime.now()
            c0_c1_copy_time = c0_c1_cp_end_time - c0_c1_cp_start_time
            split_id += 1
            print(split_id,seq1_time, c0_c1_copy_time, seq2_time, seq_fc_time, mb_device_0_start_time, mb_device_0_end_time, c0_c1_cp_start_time, c0_c1_cp_end_time, mb_device_1_start_time, mb_device_1_end_time, mb_fc_start_time, mb_fc_end_time, mb_device_0_start_datetime, mb_device_0_end_datetime, c0_c1_cp_start_datetime, c0_c1_cp_end_datetime, mb_device_1_start_datetime, mb_device_1_end_datetime, mb_fc_start_datetime, mb_fc_end_datetime)


        mb_device_1_start_time = time.time()
        s_prev = self.seq2(s_prev)
        mb_device_1_start_datetime = datetime.now()
        mb_device_1_end_time = time.time()
        mb_device_1_end_datetime = datetime.now()
        seq2_time = mb_device_1_end_time - mb_device_1_start_time
        mb_fc_start_time = time.time()
        mb_fc_start_datetime = datetime.now()
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
        mb_fc_end_time = time.time()
        mb_fc_end_datetime = datetime.now()
        seq_fc_time = mb_fc_end_time - mb_fc_start_time
        split_id += 1
        print(split_id,seq1_time, c0_c1_copy_time, seq2_time, seq_fc_time, mb_device_0_start_time, mb_device_0_end_time, c0_c1_cp_start_time, c0_c1_cp_end_time, mb_device_1_start_time, mb_device_1_end_time, mb_fc_start_time, mb_fc_end_time, mb_device_0_start_datetime, mb_device_0_end_datetime, c0_c1_cp_start_datetime, c0_c1_cp_end_datetime, mb_device_1_start_datetime, mb_device_1_end_datetime, mb_fc_start_datetime, mb_fc_end_datetime)

        return torch.cat(ret)


num_batches = 3
batch_size = 120
image_w = 128
image_h = 128

import time


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for _ in range(num_batches):
        print("------------ Minibatch {} --------------".format(_))
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
            .scatter_(1, one_hot_indices, 1)

        # run forward pass
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        t1 = time.time()
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))
        fw_time = time.time() - t1
        #print("Output-device {}".format(outputs.device))

        # run backward pass
        t1 = time.time()
        labels = labels.to(outputs.device)
        label_copy_time = time.time() -t1
        t1 = time.time()
        loss_fn(outputs, labels).backward()
        bw_time = time.time() - t1
        t1 = time.time()
        optimizer.step()
        opt_time = time.time() - t1

        #print(prof.key_averages().table(sort_by="cuda_time"))    

        print("FW {}, LBL_CP {}, BW {}, OPT {}".format(fw_time, label_copy_time, bw_time, opt_time))


#########
print("Running Model Parallel Resnet50")
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 1

stmt = "train(model)"

########### Pipeline Parallel ################
print("Running Pipeline Parallel ResNet50")

#setup = "model = PipelineParallelResNet50()"
#pp_run_times = timeit.repeat(
#    stmt, setup, number=1, repeat=num_repeat, globals=globals())
#pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)


# plot([mp_mean, rn_mean, pp_mean],
#      [mp_std, rn_std, pp_std],
#      ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
#      'mp_vs_rn_vs_pp.png')


##### Variable Split Sizes for Batch #####
print("Running Pipeline Parallel ResNet50 for multiple split sizes")
means = []
stds = []
split_sizes = [1, 2, 4, 5, 10, 20, 50, 100]
#split_sizes = [1, 3, 5, 8, 10, 12, 20, 40, 60]
#split_sizes = [2, 4, 8, 10, 12, 20, 40, 60]
#split_sizes = [10, 12, 20, 40, 60]
split_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

for split_size in split_sizes:
    print("###############################################")
    print("Split Size {}".format(split_size))
    print("###############################################")
    setup = "model = PipelineParallelResNet50(split_size=%d)" % split_size
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    means.append(np.mean(pp_run_times))
    stds.append(np.std(pp_run_times))

###########################################

#print("Pipeline Mean {} ".format(pp_mean))
print("Pipeline Variables : {}".format(means))
