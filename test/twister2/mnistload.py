import os
import numpy as np
from numpy import genfromtxt

__base_path = "/tmp/twister2deepnet/mnist"
__train_base_path = "train"
__test_base_path = "test"

__train_data_file = "train-images-idx3-ubyte.npy"
__train_target_file = "train-labels-idx1-ubyte.npy"
__test_data_file = "t10k-images-idx3-ubyte.npy"
__test_target_file = "t10k-labels-idx1-ubyte.npy"

train_files = os.listdir(os.path.join(__base_path, __train_base_path))
test_files = os.listdir(os.path.join(__base_path, __test_base_path))

__train_data_file_path = os.path.join(__base_path, __train_base_path, __train_data_file)
__train_target_file_path = os.path.join(__base_path, __train_base_path, __train_target_file)

__test_data_file_path = os.path.join(__base_path, __test_base_path, __test_data_file)
__test_target_file_path = os.path.join(__base_path, __test_base_path, __test_target_file)

training_data = None
training_target = None
testing_data = None
testing_target = None

if os.path.exists(__train_data_file_path) and os.path.exists(__train_target_file_path):
    print("Training data available")
    training_data = np.load(__train_data_file_path)
    training_target = np.load(__train_target_file_path)

if os.path.exists(__test_data_file_path) and os.path.exists(__test_target_file_path):
    print("Testing data available")
    testing_data = np.load(__test_data_file_path)
    testing_target = np.load(__test_target_file_path)

if training_data is not None and training_target is not None:
    print(training_data.shape, training_target.shape)
    np.savetxt(__train_data_file_path.split(".")[0] + ".csv", training_data, delimiter=',')
    np.savetxt(__train_target_file_path.split(".")[0] + ".csv", training_target, delimiter=',')

if testing_data is not None and testing_target is not None:
    print(testing_data.shape, testing_target.shape)
    np.savetxt(__test_data_file_path.split(".")[0] + ".csv", testing_data, delimiter=',')
    np.savetxt(__test_target_file_path.split(".")[0] + ".csv", testing_target, delimiter=',')
