# PoseNet_Planar
In this project, we investigate efficient small-sized convolutional neural networks (CNNs) for ego-motion estimation of micro air vehicles (MAVs) equipped with a downward-facing camera and inertial measurement unit (IMU), especially focusing on the robustness of estimation in fast maneuvers. 

# Publication
CNN-based Ego-Motion Estimation for Fast MAV Maneuvers (accepted by ICRA 2021)

Article (with Appendix): https://arxiv.org/abs/2101.01841

Videos: https://youtu.be/BMdh6dmLgrM https://youtu.be/Uz9pNpn94jU

# Prerequisities
```
Python3 (tested with v3.6 and v3.8) 
PyTorch v1.1.0 (tested with v1.1.0 and v1.4.0)
Python packages: cv2, numpy, torchvision, tensorboardX.
MATLAB (tested with v2019b and v2020a)
```

# Usage
## 1.Dataset Generation 
(MATLAB) image_generation_matlab/main.m

There are 5 image pairs in the example_images folder. The code uses the example image pairs by default. To use your own dataset, change the directory in the params.py file.

## 2.Network Training
Self-supervised learning (default): 
```bash
python3 main.py
```
Supervised learning: 
```bash
python3 main.py --supervised
```
## 3.Pre-trained Models
We provide pre-trained models (self-supervised learning) for 4 different network architectures in the pretrained_models folder. Check the paper and params.py file for details. To train with a pre-trained model and show the photometric errors before (derotated) and after (warped with the network prediction):
```bash
python3 main.py --use-pretrained --show-img
```
To test the pre-trained models on testing set and show the photometric errors:
```bash
python3 main.py --test  --show-img
```
Press any key to continue when using --show-img

## 4.Running on Public Flight Dataset
Use the following command to run the network on indoor 45 degree downward facing sequences of the UZH-FPV dataset (https://fpv.ifi.uzh.ch/datasets/) and show the photometric errors. Change the directory to the dataset on your own device in file run_UZHFPV.py.
```bash
python3 main.py --flight --show-img
```
<img src='https://github.com/YingfuXu/PoseNet_Planar/blob/main/example_images/photometric_error_example/before.png' width=256>  <img src='https://github.com/YingfuXu/PoseNet_Planar/blob/main/example_images/photometric_error_example/after.png' width=256>

# Citation

If you use this repository in an academic context, please cite the following:
```
@article{xu2021cnn,
  title={CNN-based Ego-Motion Estimation for Fast MAV Maneuvers},
  author={Xu, Yingfu and de Croon, Guido CHE},
  journal={arXiv preprint arXiv:2101.01841},
  year={2021}
}
```

# Authors and Maintainers
Yingfu Xu (y.xu-6@tudelft.nl) and Prof. Dr. Guido de Croon from Micro Air Vehicle Laboratory (MAVLab), TU Delft, The Netherlands.

# Acknowledgement

We greatly thank Ir. Nilay Y. Sheth (https://github.com/nilay994) for his supports in developing the quadrotor MAV with GPU and collecting the flight datasets. We also appreciate the author of https://github.com/ClementPinard/FlowNetPytorch upon the code structure of which we developed our code for network training. The Madgwick AHRS code in this repo is taken from https://github.com/morgil/madgwick_py, whose author is appreciated as well.
