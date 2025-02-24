
# A Metric to Assess Neural Network Resilience Against Adversarial Samples

This is the code repository to the paper "An Analytical Dive into Finding a Metric to Assess Neural Network Resilience Against Adversarial Samples". The results of the evaluation in the paper can be found in detail in folder __"metric_results"__.

When using the metric please cite:
```

```

As the implemented attacks were from the "__TorchAttacks__" package, please also cite:
```
@article{kim2020torchattacks,
title={Torchattacks: A pytorch repository for adversarial attacks},
author={Kim, Hoki},
journal={arXiv preprint arXiv:2010.01950},
year={2020}
}
```

__Contents:__
- __"data_zips"__: Zipped data used for the example in the paper (Refer to _Extracting the data: "decode_torchvision_data.py"_)
- __"metric_code"__: Code and models used for evaluation in paper (Refer to _Usage_)
- __"metric_results"__: Results presented in the paper
- __"requirements.txt"__: Used packages (Refer to _Installation_)


## Installation

The code was written with Python 3.11 and uses the packages in "requirements.txt".

```bash
  python -m pip install -r "requirements.txt"
```
    
## Usage

The following section describes how the presented code was used for the evaluation results in the paper. With this the functionality of the different scripts is described.

### Extracting the data: "decode_torchvision_data.py"
In order to better handle the different necessary datasets from the CIFAR10-dataset (https://www.cs.toronto.edu/~kriz/cifar.html) the compressed structure is read and every image is saved as a .png-file with the following naming convention:

```
index_"number1"_true_"number2"_target_"number3".png
```
- "number1" is a integer which determins the index of the picture. Every seperate dataset needs to be indexed starting from 0. (For that the "rename_indices.py" script can help)
- "number2" is an integer which gives the correct class label of the image, starting from class 0 to class 9.
- "number3" is an integer which stores a target label for a targeted adversarial sample. The generation algorithm will try to form an adversarial sample with this class (For that the randomize_targets.py" script can help). For a untargeted sample the target will be set to -1. For training or testing the target value is not used.

With every datapoint stored as a .png-file we can manually separate, move or copy the files to the desired locations. For example as described in the paper. 

For training a "test" and "train" folder is necessary, for the substitute data generation a start-data folder is necessary and for the metric calculation a "targeted"- and "untargeted"-data folder is necessary.

The used data for the evaluation in the paper is stored in the zip-files in "data_zips". Note that the "train.zip" and "train1.zip" data needs to merged in order to form the original "train"-data.

#### Usage of "decode_torchvision_data.py"
- _data_path_: Path to the folder where the original CIFAR10 data will be downloaded to.
- _output_path_: Path to where the "train" and "test" folder with the .png-files will be written.

### Setting the right indices: "rename_indices.py"
As described every folder containing data needs to have indices starting at 0. This script renames the datapoints to start with the correct indices.

#### Usage of "rename_indices.py"
- _data_path_: Path to the folder containing the incorrectly indexed .png-files.

### Randomizing target labels: "randomize_targets.py"
In order to perform a targeted attack, every target needs to be set to a random value unequal to the true label. This script randomizes these targets accordingly.

#### Usage of "randomize_targets.py"
- _data_path_: Path to the folder containing the .png-files which need to get random targets.

### Training a white box model: "network_training.py"
Now a white box model can be trained. The architecture of the model can be written as a class in a seperate file an imported into the code at line 139. All the hyperparameters can be set from line 115 aswell. 

The script runs the training with the set hyperparameters on a cuda-device if available. In the end a .pth-checkpoint will be stored and a result-graph created. 

#### Usage of "network_training.py"
- _data_path_: Path to a folder containing a "train"-folder with the images for training and a "test"-folder with the images for testing.
- _working_dir_: Folder to where the checkpoint and the result-graphs will be stored.
- _model_name_: Name of the checkpoint-File (Without file-ending).

Note that the corresponing architecture of the model must be imported at line 139.

### Performing a Grey-Box or Black-Box attack ("label_substitude_data.py", "train_substitute_network.py", "augment_substitute_data.py")
In order to train a grey- or black-box model for a attack, a substitude dataset must be created. These three scripts are used for that. 

First a folder containing a small amount of datapoints for the start of the algorithm is needed (These can be 10-100 images). Next these images are labled by the white-box model with the "label_substitude_data.py" script. Then the network can be trained with the "train_substitute_network.py" script. If the performance is acceptable, the process can stop, else the dataset is augmented with the "augment_substitute_data.py" script. Then the process is repeated until performance is acceptable.

#### Usage of "label_substitude_data.py"
- _target_network_path_: Path to the checkpoint of the white-box model.
- _substitude_data_path_: Path to the folder containing the current substitude dataset (Can also be the start-dataset).

Note that the corresponing architecture of the white-box model must be imported at line 34.

#### Usage of "train_substitute_network.py"
- _substitude_data_path_: Path to a folder containing the data generated for training.
- _working_dir_: Folder to where the checkpoint and the result-graphs will be stored.
- _model_name_: Name of the checkpoint-File (Without file-ending).

Note that the corresponing architecture of the grey-box or black-box model must be imported at line 62.

#### Usage of "augment_substitute_data.py"
- _substitude_data_path_: Path to a folder containing the augmented data.
- _target_network_path_: Path to the checkpoint of the white-box model.
- _step_size_: Hyperparameter of the augmentation algorithm.

Note that the corresponing architecture of the white-box model must be imported at line 44.

### Calculating the Metric: "metric_calculation.py"
Now the metric can be calculated as a white-box, grey-box and black-box model have been trained and a corresponing data-folder for targeted and untargeted attacks has be created. The metric will store its outputs in a .pkl-file which can be read by the "print_metric.py" file. Note that the metric will store every generated adversarial sample in the stated folder.

#### Usage of "metric_calculation.py"
- _task_name_: Name of the folder to store the result.
- _white_box_model_path_: Path to the checkpoint of the white-box model.
- _grey_box_model_path_: Path to the checkpoint of the grey-box model.
- _black_box_model_path_: Path to the checkpoint of the black-box model.
- _targeted_data_path_: Path to the folder containing the images to use for the targeted attack.
- _untargeted_data_path_: Path to the folder containing the images to use for the untargeted attack.
- _print_metric_to_png_: If True the "print_metric.py" will be automaticly executed after the metric calculation.
- _distance_limit_max_: Maximum L2-distance limitation for the last metric run.
- _distance_limit_min_: Minimum L2-distance limitation for the first metric run.
- _distance_steps_: Amount of metric runs from min to max.
- _max_iterations_: Maximum amount of iterations one attack can run.

Note that the corresponing architectures of the white-box, grey-box and black-box model must be imported at line 65, 72, 79.

### Trying a defense-strategy: "adversarial_training.py"
For the evaluation in the paper four different types of adversarial training were used. This script was used to train the models. Functionality is the same as "network_training.py" with the addition of the adversarial training in the training-function. The type of adversarial training can be set as a parameter in line 260.

#### Usage of "adversarial_training.py"
- _data_path_: Path to a folder containing a "train"-folder with the images for training and a "test"-folder with the images for testing.
- _working_dir_: Folder to where the checkpoint and the result-graphs will be stored.
- _model_name_: Name of the checkpoint-File (Without file-ending).

In line 260: "generation_algo_mode":
- 0: PGD adversarial training
- 1: CW adversarial training
- 2: PGD & CW adversarial training
- 3: DeepFool adversarial training

Note that the corresponing architecture of the model must be imported at line 253.
