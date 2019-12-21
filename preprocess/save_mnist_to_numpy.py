import numpy as np
from torchvision import datasets, transforms

train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

train_set_array_data = train_set.data.numpy().astype(np.float32)
test_set_array_data = test_set.data.numpy().astype(np.float32)
train_set_array_target =train_set.targets.numpy()
test_set_array_target =test_set.targets.numpy()

np.save("datasets/train_data.npy", train_set_array_data)
np.save("datasets/test_data.npy", test_set_array_data)

np.save("datasets/train_target.npy", train_set_array_target)
np.save("datasets/test_target.npy", test_set_array_target)

#print(train_set_array[0])