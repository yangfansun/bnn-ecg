# *ECG Signal Classification with Binarized Convolutional Neural Network* source code

This code is the source code of our [paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482520301682). Please refer to the paper for the specific parameters.

### Dataset

The dataset we used for model verification is from the [PhysioNet/CinC](https://physionet.org/content/challenge-2017/1.0.0/) arrhythmia detection challenge 2017, which contains 8,528 single-lead ECG recordings, each of which is derived from participants with length ranging from 9 seconds to 61 seconds. The data were sampled at 300 Hz (the shortest data has 2,714 data points and the longest data has 18,286).

Please using `read_data.py` to create the `tfrecoed` format data.

### Train

You can edit the file `myscript.sh` to adjust the hyper parameters of the training, and run `entry.py` to start running the program.

### Test

Set the `is_test` to `True` in `myscrips.sh` to test the model.

### Dependent Library

python==2.7
tensorflow==1.5.0
tensorlayer==1.8.3
CUDA==9.1
cudnn==7.3.1
