import numpy as np
# import matplotlib.pyplot as plt
import sys
import os
import time
import cv2
import torchvision.transforms as transforms
import torch
import math
from csv import reader

from util import absolutePose2homo8Pose
from ahrs import *

# NOTE change the dataset directory and the number of sequence to run here
dataset_folder = '/home/yingfu/datasets/UZH-FPV/' # /home/adr/datasets/UZHFPV/
flight_environment = 'indoor' # outdoor / indoor 
sequence_num = '2' # 1 / 2 4 9 12 13 14

# whether to use the prior pose
use_prior_pose = False #  False True
imwrite = False
# equalizeHist = False

if flight_environment == 'outdoor':
    sequence = 'outdoor_45_' + sequence_num + '_snapdragon_with_gt'
    cameraMatrixUZHFPV = np.array([[275.3385453506587, 0, 315.7697752181792],
                                    [0, 275.0852058534152, 233.72625444124952],
                                    [0, 0, 1]])
    distCoeffsUZHFPV = np.array([-0.017811595366268803, 0.04897078939103475, -0.041363300782847834, 0.011440891936886532]) # fish eye

else:
    sequence = 'indoor_45_' + sequence_num + '_snapdragon_with_gt'
    cameraMatrixUZHFPV = np.array([[275.46015578667294, 0, 315.958384100568],
                                    [0, 274.9948095922592, 242.7123497822731],
                                    [0, 0, 1]])
    distCoeffsUZHFPV = np.array([-6.545154718304953e-06, -0.010379525898159981, 0.014935312423953146, -0.005639061406567785]) # fish eye

results_save_path = dataset_folder+sequence+'/PoseNet_Planar_results/'

R2D = 180.0/math.pi
FoV = 45.0*2
networkCameraMatrix = np.array([[(320.0-1.0)/2.0/math.tan(FoV/180*3.14159265/2), 0, (320.0-1.0)/2.0*1],
                [0, (320.0-1.0)/2.0/math.tan(FoV/180*3.14159265/2), (224.0-1.0)/2.0*1],
                [0, 0, 1]])
undist_map1, undist_map2 = cv2.fisheye.initUndistortRectifyMap(cameraMatrixUZHFPV, distCoeffsUZHFPV, None, networkCameraMatrix, (320, 224), cv2.CV_32FC1) # print(map1, map2)

T_cam_imu = np.array([
    [-0.027256691772188965, -0.9996260641688061, 0.0021919370477445077, 0.02422852666805565],
    [-0.7139206120417471, 0.017931469899155242, -0.6999970157716363, 0.008974432843748055],
    [0.6996959571525168, -0.020644471939022302, -0.714142404092339, -0.000638971731537894],
    [0.0, 0.0, 0.0, 1.0]
])

RotMtrx_cam_imu = T_cam_imu[0:3, 0:3] # this mtrx rotate a vector from IMU frame to cam frame, NOTE IMU mounted as front-left-up
RotMtrx_imu_body = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) # body frame: front-right-down
RotMtrx_cam_body = np.dot(RotMtrx_cam_imu, RotMtrx_imu_body)

imu_file = dataset_folder+sequence+'/imu.txt'
img_stream_file = dataset_folder+sequence+'/left_images.txt'
img_folder = dataset_folder+sequence+'/img/'
undistort_img_save_folder = dataset_folder+sequence+'/undistorted_img_left/'

if imwrite and not os.path.exists(undistort_img_save_folder):
    os.makedirs(undistort_img_save_folder)

# roughly calculate the bias of gyro by averaging the measurements when the drone is stationary
if sequence_num == '2':
    bias_gyro = np.array([-0.0063532372, 0.00788615, 0.0032384]) # 
    # bias_acc = np.array([-0.16805294, 0.04594449, -0.1789313259]) # 
    bias_acc = np.array([-0.0, 0.0, -0.0])
elif sequence_num == '4':
    bias_gyro = np.array([-0.006796387, 0.00665151, 0.00474788368]) # 
    # bias_acc = np.array([-0.1719411036, 0.0604868, -0.05244188976]) # 
    bias_acc = np.array([-0.0, 0.0, -0.0])
elif sequence_num == '9':
    bias_gyro = np.array([-0.0039169774, 0.0081098583, 0.00123357624]) # 
    # bias_acc = np.array([-0.1128046, 0.0798941, -0.3837015]) # 
    bias_acc = np.array([-0.0, 0.0, -0.0])
elif sequence_num == '12':
    bias_gyro = np.array([-0.00464348776, 0.0069274, 0.003343865]) # 
    # bias_acc = np.array([0.0251198348, -0.016069799, -0.1670273895263]) # 
    bias_acc = np.array([-0.0, 0.0, -0.0])
if sequence_num == '13':
    bias_gyro = np.array([-0.0039052595, 0.0041236387, 0.00317235755]) # 
    # bias_acc = np.array([-0.0143890781, 0.015423367847, -0.0982426014]) #
    bias_acc = np.array([-0.0, 0.0, -0.0])
elif sequence_num == '14':
    bias_gyro = np.array([-0.0038903458, 0.00361550756, 0.00328527558]) # 
    # bias_acc = np.array([-0.021202941, 0.0058609755384, -0.0842127504349]) #  
    bias_acc = np.array([-0.0, 0.0, -0.0])
elif sequence_num == '1':
    bias_gyro = np.array([-0.024039822921, -0.0013624732392, 0.010666493]) # 
    # bias_acc = np.array([-0.7924718946218211, -0.123195408187832, 0.023751075744671724]) #  
    bias_acc = np.array([-0.0, 0.0, -0.0])
else:
    bias_gyro = np.array([0.0, 0.0, 0.0]) # 
    bias_acc = np.array([-0.0, 0.0, -0.0])

timeshift_cam_imu = -0.01484888826656275


def flight_dataset_test(posenet_model, device):

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    grayscale_input_transform = transforms.Compose([
        ArrayToTensor(), # transform numpy array to PyTorch tensor
        transforms.Normalize(mean=[0], std=[255])
    ])

    # go through stream.txt to find the correspondences between image file names and timestamps and then save them into a list
    data = reader(open(img_stream_file))
    list_time_img = []
    for index, row in enumerate(data):
        if index > 0: # skip the first row
            timestamp = (row[0].split(' ')[1])
            seq = int(row[0].split(' ')[0])
            img_file_name = row[0].split(' ')[2].split('/')[1]
            list_time_img.append([float(timestamp), img_file_name])

    ahrs = madgwickahrs.MadgwickAHRS(quaternion=np.array([1.0, 0.0, 0.0, 0.0]), beta=0.0005)

    Euler_ahrs_list = []
    IMU_list = []
    nn_distScaledVelBody_list = []
    initialize_flag_img1_angle = 0
    prior_pose_trans = np.array([0.0, 0.0, 0.0])
    img_index = 101 # start from the 100th pair
    img_stride = 1 # hard to get good result when img_stride > 1, UZH-FPV dataset is too fast.

    f1 = open(imu_file,"r")
    lines = f1.readlines()[1:]  # skip the first line

    initialTimeStamp = float(lines[0].split(' ')[1])
    print('Initial Time Stamp:', initialTimeStamp) # IMU first measurement
    timeStampOld = initialTimeStamp

    # imu data main loop
    for line in lines: # imu data main loop
        timeStamp = float(line.split(' ')[1])
        wx = float(line.split(' ')[2])
        wy = -float(line.split(' ')[3]) # minus sign to transfer the measurement to the body frame
        wz = -float(line.split(' ')[4])
        ax = float(line.split(' ')[5])
        ay = -float(line.split(' ')[6])
        az = -float(line.split(' ')[7])
        dt = timeStamp - timeStampOld

        # print(dt,timeStamp,wx,wy,wz,ax,ay,az)
        ahrs.update_imu(dt, [wx-bias_gyro[0],wy-bias_gyro[1],wz-bias_gyro[2]], [-(ax-bias_acc[0]),-(ay-bias_acc[1]),-(az-bias_acc[2])])
        roll = ahrs.quaternion.to_Euler_ZYX()[0] # body frame
        pitch = ahrs.quaternion.to_Euler_ZYX()[1]
        yaw = ahrs.quaternion.to_Euler_ZYX()[2]
        Euler_ahrs_list.append([timeStamp, roll, pitch, yaw])
        IMU_list.append([timeStamp, wx, wy, wz, ax, ay, az])
        # print(timeStamp, roll, pitch, yaw)

        timeStampOld = timeStamp # imu time

        if img_index > len(list_time_img)-1:
            print("Flight sequence has ended.")
            break

        img1_timeStamp = list_time_img[img_index-img_stride][0] + timeshift_cam_imu
        img2_timeStamp = list_time_img[img_index][0] + timeshift_cam_imu

        if timeStamp > img1_timeStamp and initialize_flag_img1_angle == 0: # only for the first image

            print("First image timeStamp and the IMU timeStamp:", img1_timeStamp, timeStamp)

            img_roll, img_pitch, img_yaw = Rot_to_Euler(np.dot(RotMtrx_cam_body, ahrs.quaternion.to_Rot_Mtrx()))
            # print(roll*R2D, pitch*R2D, yaw*R2D)

            img1_roll = img_roll
            img1_pitch = img_pitch
            img1_yaw = img_yaw
            initialize_flag_img1_angle = 1

        if timeStamp > img2_timeStamp: # feed image pair into the network

            img_roll, img_pitch, img_yaw = Rot_to_Euler(np.dot(RotMtrx_cam_body, ahrs.quaternion.to_Rot_Mtrx())) # Euler angles of the camera

            img2_roll = img_roll
            img2_pitch = img_pitch
            img2_yaw = img_yaw

            dt_img = img2_timeStamp - img1_timeStamp
            # print('dt_img:',dt_img)

            img1_file = img_folder + list_time_img[img_index-img_stride][1]
            img2_file = img_folder + list_time_img[img_index][1]

            img1_Euler = np.array([img1_roll, img1_pitch, img1_yaw])
            img2_Euler = np.array([img2_roll, img2_pitch, img2_yaw])

            print(img1_file.split('/')[-1])

            # print("prior_pose_trans =", prior_pose_trans)
            nn_pred = image_pair_posenet(grayscale_input_transform, img1_file, img2_file, img1_Euler, img2_Euler, prior_pose_trans, posenet_model, device)
            # nn_pred = np.array([1, 1, 1]) # for debugging Euler
            
            if use_prior_pose:
                prior_pose_trans[0] = nn_pred[0]
                prior_pose_trans[1] = nn_pred[1]
                prior_pose_trans[2] = nn_pred[2]
            
            velocityCamOverDist = nn_pred / dt_img # expressed in the img2 frame
            distScaledVelBody = np.dot(np.transpose(RotMtrx_cam_body), np.array([[velocityCamOverDist[0]], [velocityCamOverDist[1]], [velocityCamOverDist[2]]]))
            nn_distScaledVelBody_list.append([timeStamp, distScaledVelBody[0], distScaledVelBody[1], distScaledVelBody[2]])
            
            print("Distance-scaled Velocity (Body frame) :", timeStamp, distScaledVelBody.reshape(3))
            print(" ")

            img1_roll = img_roll
            img1_pitch = img_pitch
            img1_yaw = img_yaw

            img_index = img_index + 1


    IMU = np.array(IMU_list)

    # # bias rough calculation
    # print(np.mean(IMU[0:1000, 1]), np.mean(IMU[0:1000, 2]), np.mean(IMU[0:1000, 3]))
    # print(np.mean(IMU[0:1000, 4]), np.mean(IMU[0:1000, 5]), np.mean(IMU[0:1000, 6])+9.81)

    Euler_ahrs = np.array(Euler_ahrs_list) # timeStamp, roll, pitch, yaw

    ahrs_save_path = results_save_path+'Euler_ahrs_array.txt'
    np.savetxt(ahrs_save_path, Euler_ahrs)

    nn_distScaledVelBody = np.array(nn_distScaledVelBody_list)

    if use_prior_pose:
        np.savetxt(results_save_path+'nn_distanceScaledVelocity_prior.txt', nn_distScaledVelBody)
    else:
        np.savetxt(results_save_path+'nn_distanceScaledVelocity.txt', nn_distScaledVelBody)

    # plt.figure(1) # plot for IMU and AHRS
    # plt.subplot(311)
    # roll = plt.plot(Euler_ahrs[:, 0], Euler_ahrs[:, 1]*R2D,'r',label='ahrs_roll') 

    # gyrox = plt.plot(IMU[:, 0], IMU[:, 1],'b',label='gyrox') 
    # accx = plt.plot(IMU[:, 0], IMU[:, 4],'g',label='accx') 

    # plt.title("AHRS Euler")
    # plt.xlabel('time')
    # plt.ylabel('roll/deg')

    # plt.legend()
    # plt.subplot(312)
    # pitch = plt.plot(Euler_ahrs[:, 0], Euler_ahrs[:, 2]*R2D,'r',label='ahrs_pitch') 
    
    # gyroy = plt.plot(IMU[:, 0], IMU[:, 2],'b',label='gyroy') 
    # accy = plt.plot(IMU[:, 0], IMU[:, 5],'g',label='accy') 
    
    # plt.xlabel('time')
    # plt.ylabel('pitch/deg')

    # plt.legend()
    # plt.subplot(313)
    # yaw = plt.plot(Euler_ahrs[:, 0], Euler_ahrs[:, 3]*R2D, 'r',label='ahrs_yaw')

    # gyroz = plt.plot(IMU[:, 0], IMU[:, 3],'b',label='gyroz') 
    # accz = plt.plot(IMU[:, 0], IMU[:, 6],'g',label='accz') 

    # plt.xlabel('time')
    # plt.ylabel('yaw/deg')

    # plt.legend()
    # plt.show()


def image_pair_posenet(grayscale_input_transform, img1_file, img2_file, img1_Euler, img2_Euler, prior_pose_trans, posenet_model, device):

    undist_img1 = cv2.remap(cv2.imread(img1_file, cv2.IMREAD_GRAYSCALE).astype(np.float32), undist_map1, undist_map2, cv2.INTER_LINEAR)
    undist_img2 = cv2.remap(cv2.imread(img2_file, cv2.IMREAD_GRAYSCALE).astype(np.float32), undist_map1, undist_map2, cv2.INTER_LINEAR)
    
    if imwrite:
        cv2.imwrite(undistort_img_save_folder+img1_file.split('/')[-1], undist_img1.astype(np.uint8))

    # if equalizeHist:
    #     undist_img1 = cv2.equalizeHist(undist_img1.astype(np.uint8))
    #     undist_img2 = cv2.equalizeHist(undist_img2.astype(np.uint8))

    img1 = grayscale_input_transform(undist_img1).unsqueeze(0).to(device) # torch.Size([1, 1, 224, 320])
    img2 = grayscale_input_transform(undist_img2).unsqueeze(0).to(device)

    img1_Euler_tensor = torch.from_numpy(img1_Euler).float()
    pose_img1 = torch.cat((img1_Euler_tensor, torch.zeros(3)),dim=0).unsqueeze(0)

    img2_Euler_tensor = torch.from_numpy(img2_Euler).float()
    pose_img2 = torch.cat((img2_Euler_tensor, torch.zeros(3)),dim=0).unsqueeze(0)

    _,homo8_1to2_initial = absolutePose2homo8Pose(pose_img1, pose_img2, batch_size=1, rot_random_bias=0.0, slope_random_bias=0.0)

    homo8_1to2_initial[0, 5] = prior_pose_trans[0]
    homo8_1to2_initial[0, 6] = prior_pose_trans[1]
    homo8_1to2_initial[0, 7] = prior_pose_trans[2]

    before_net = time.time()
    output = posenet_model(img1,img2,homo8_1to2_initial.to(device))
    print("Network run time (Hz):", 1 / (time.time() - before_net))
    # output = [torch.ones(1, 8)] # for purely saving undistorted images (speed up by not running the network)

    return np.array([output[0][0, 5].to('cpu'), output[0][0, 6].to('cpu'), output[0][0, 7].to('cpu')])


def Rot_to_Euler(RotMtrx):

    rot23 = RotMtrx[1, 2]
    rot33 = RotMtrx[2, 2]
    rot32 = RotMtrx[2, 1]
    rot22 = RotMtrx[1, 1]
    rot12 = RotMtrx[0, 1]
    rot11 = RotMtrx[0, 0]
    rot13 = RotMtrx[0, 2]
    sy = np.sqrt(rot23*rot23 + rot33*rot33)
    if sy < 1e-6:
        print("Pitch is close to 90 degrees! ")
        yaw = 0.0
        roll = np.arctan2(-rot32, rot22)
    else:
        yaw = np.arctan2(rot12, rot11)
        roll = np.arctan2(rot23, rot33)
    pitch = np.arctan2(-rot13, sy)
    return roll, pitch, yaw


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        
        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.float()
