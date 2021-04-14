import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import time
from torch.nn.init import kaiming_normal_, xavier_uniform_, zeros_

import warp
from util import *

# __all__ = [
#     'ICSTN_FPE' # feature pyramid extractor (FPE) 
# ]

def conv(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
        nn.ReLU(inplace=True)
    )

class FPE_Net(nn.Module):
    
    def __init__(self, list_blocks_types, img_height, img_width, device, self_supervised, normalization=None, 
                 share_fc_weights=True, fc_dropout_rate=0.0, show_img=False, trace_model = False):

        super().__init__() 

        self.device = device
        self.max_pyramid_level = len(list_blocks_types) + 1 # each block corresponds to one pyramid level, '+1' to correspond to Table II - 6th
        self.num_blocks = len(list_blocks_types)
        self.show_img = show_img 
        self.share_fc = share_fc_weights
        self.trace_model = trace_model
        self.self_sup = self_supervised

        self.img_warper_dict = {}
        # initialize the image warpers for each pyramid_level. 3 warpers for the feature maps (multi-channel) and 1 warper for the image (for img_show)
        for pyramid_level in range(self.max_pyramid_level): # NOTE for now, each pyramid level has same posenet blocks
            print("Initializing PoseNet and ImgWarper for pyramid level {} ...".format(pyramid_level)) 
            input_tensor_height = int(img_height/(2**pyramid_level))
            input_tensor_width = int(img_width/(2**pyramid_level))
            self.img_warper_dict[pyramid_level] = warp.WarpImg(img_height=input_tensor_height, img_width=input_tensor_width, device=self.device)

        self.dof = [] 
        for motion in list_blocks_types:
            if motion == 'trans' or motion == 'rot': 
                self.dof.append(3) # NOTE degree-of-freedom to predict (for now translation only)
            elif motion == 'tilt':
                self.dof.append(2)     
            elif motion == 'rot&trans':
                self.dof.append(6)

        if self.show_img:
            cv2.namedWindow('before', cv2.WINDOW_NORMAL)
            cv2.namedWindow('after', cv2.WINDOW_NORMAL)

        conv_planes = [16, 32, 64, 128, 256]
        linear_inputs = [5120, 5120, 5120]

        # feature pyramid extractor (acts on each image respectively) (Table II - 6th)
        self.conv1 = conv(             1, conv_planes[0], kernel_size=7, stride=4) # 56 80  
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2) # 28 40  
        self.conv3 = conv(conv_planes[1], conv_planes[2], stride=2) # 14 20  

        # Pose prediction blocks
        self.convs_block_1 = nn.Sequential( # 14 20  
                conv(conv_planes[2]*2, conv_planes[3], stride=2), # 7 10
                conv(conv_planes[3], conv_planes[4], stride=2) # 4 5
                )
        self.linear_input_1 = linear_inputs[0]
        self.fc_block_1 = nn.Linear(self.linear_input_1, self.dof[0], bias=True)

        self.convs_block_2 = nn.Sequential(# 28 40 
                conv(conv_planes[1]*2, conv_planes[2], stride=2), # 14 20
                conv(conv_planes[2], conv_planes[3], stride=2), # 7 10
                conv(conv_planes[3], conv_planes[4], stride=2) # 4 5
                )
        if self.share_fc:
            assert(linear_inputs[1] == linear_inputs[0])
            self.fc_block_2 = self.fc_block_1
        else:
            self.linear_input_2 = linear_inputs[1]
            self.fc_block_2 = nn.Linear(self.linear_input_2, self.dof[1], bias=True)
        
        self.convs_block_3 = nn.Sequential( # 56 80
                conv(conv_planes[0]*2, conv_planes[1], kernel_size=5, stride=2), # 28 40
                conv(conv_planes[1], conv_planes[2], stride=2), # 14 20
                conv(conv_planes[2], conv_planes[3], stride=2), # 7 10
                conv(conv_planes[3], conv_planes[4], stride=2) # 4 5
                )
        if self.share_fc:
            assert(linear_inputs[2] == linear_inputs[0])
            self.fc_block_3 = self.fc_block_1
        else:
            self.linear_input_3 = linear_inputs[2]
            self.fc_block_3 = nn.Linear(self.linear_input_3, self.dof[2], bias=True)

        self.init_weights()

        print("FPE PoseNet has {} image pyramid levels.".format(self.num_blocks))
        print("PoseNet blocks are:", list_blocks_types)
        # print("Input fully connected:", linear_inputs)

        print("Feature Pyramids Extractor Network is initialized!")
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # zeros_(m.weight) # use pytorch default init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if m.bias is not None:
                    zeros_(m.bias)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def show_imgs_after(self, batch_img1, batch_img2, batch_trans, batch_rotMtrx, batch_planeNormVecImg1):
        batch_img2_warped = self.img_warper_dict[0].transformImage(batch_img2,batch_trans,batch_rotMtrx,batch_planeNormVecImg1)
        imgs_after = torch.abs(batch_img1 - batch_img2_warped)[0, 0, :, :]
        show_imgs = torch.cat((batch_img1[0, :, :, :], batch_img2_warped[0, :, :, :], imgs_after.unsqueeze(0)), dim=1).to('cpu')
        cv2.imshow('after', show_imgs.detach().numpy().transpose(1, 2, 0))
        cv2.waitKey(-1)

    def forward(self,batch_img1,batch_img2,batch_homo8):

        batch_size = batch_img1.size()[0]

        zeroRotMtrx = torch.eye(3).repeat(batch_size, 1, 1).to(self.device)

        # timer
        warp_time = 0.0
        net_time = 0.0

        batch_nn_trans_sum = torch.zeros(batch_size, 3).to(self.device)

        if self.trace_model:
            batch_rotMtrx, batch_planeNormVecImg1, _ = poseVec2RotMtrxAndPlaneNormVecSingle(batch_homo8) # for cpp
        else:
            batch_rotMtrx, batch_planeNormVecImg1, _ = poseVec2RotMtrxAndPlaneNormVec(batch_homo8) 
            # torch.Size([batch_size, 3, 3]) torch.Size([batch_size, 3])
        
        if not self.training:
            timer_p0 = time.time()

        # warp original image to derotate
        if self.trace_model: # for cpp
            batch_img2_deRot = self.img_warper_dict[0].transformImage_Single(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1)
        else:
            batch_img2_deRot = self.img_warper_dict[0].transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1)
    
        if self.show_img:
            imgs_before = torch.abs(batch_img1 - batch_img2_deRot)[0, 0, :, :]
            show_imgs = torch.cat((batch_img1[0, :, :, :], batch_img2_deRot[0, :, :, :], imgs_before.unsqueeze(0)), dim=1).to('cpu')
            cv2.imshow('before', show_imgs.detach().numpy().transpose(1, 2, 0))
            cv2.waitKey(-1) 

        if not self.training: # timer in inference
            deRotTime = (time.time() - timer_p0)*1000.0
            warp_time = warp_time + deRotTime
            timer_p1 = time.time()

        # run feature pyramids extractor
        feature_map1_pyr1 = self.conv1(batch_img1) # input torch.Size([batch_size, 1, 224, 320])
        feature_map1_pyr2 = self.conv2(feature_map1_pyr1)
        feature_map1_pyr3 = self.conv3(feature_map1_pyr2)

        feature_map2_pyr1 = self.conv1(batch_img2_deRot) # input torch.Size([batch_size, 1, 224, 320])
        feature_map2_pyr2 = self.conv2(feature_map2_pyr1)
        feature_map2_pyr3 = self.conv3(feature_map2_pyr2)

        # run blocks
        # block 1st
        cat_feature_maps_pyr3 = torch.cat((feature_map1_pyr3, feature_map2_pyr3), dim=1) # torch.Size([batch_size, 2, 56, 80])

        conv_out_1 = self.convs_block_1(cat_feature_maps_pyr3)
        batch_nn_trans_1 = self.fc_block_1(conv_out_1.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_1

        if self.training:
            batch_nn_blocks_list = []
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_1, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                # NOTE in self-sup, it needs to warp the original batch_img2 to get the valid_pixel_mask.
                batch_img2_warped_2,valid_pixel_mask = self.img_warper_dict[0].transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_2 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_2*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)
        else:
            netTimeBlock1 = (time.time() - timer_p1)*1000.0
            net_time = net_time + netTimeBlock1
            timer_p2 = time.time()

        ## block 2
        if self.trace_model: # for cpp
            feature_map2_pyr2_warped = self.img_warper_dict[3].transformImage_Single(feature_map2_pyr2,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        else:
            feature_map2_pyr2_warped = self.img_warper_dict[3].transformImage(feature_map2_pyr2,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)

        cat_feature_maps_pyr2 = torch.cat((feature_map1_pyr2, feature_map2_pyr2_warped), dim=1)
            
        if not self.training: # timer in inference
            warpTimeBlock2 = (time.time() - timer_p2)*1000.0
            warp_time = warp_time + warpTimeBlock2
            timer_p2 = time.time()
        
        conv_out_2 = self.convs_block_2(cat_feature_maps_pyr2)
        batch_nn_trans_2 = self.fc_block_2(conv_out_2.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_2

        if self.training:
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_2, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_3,valid_pixel_mask = self.img_warper_dict[0].transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_3 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_3*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)

        else:
            netTimeBlock2 = (time.time() - timer_p2)*1000.0
            net_time = net_time + netTimeBlock2
            timer_p3 = time.time()

        ## block 3
        if self.trace_model: # for cpp
            feature_map2_pyr1_warped = self.img_warper_dict[2].transformImage_Single(feature_map2_pyr1,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        else:
            feature_map2_pyr1_warped = self.img_warper_dict[2].transformImage(feature_map2_pyr1,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)

        cat_feature_maps_pyr1 = torch.cat((feature_map1_pyr1, feature_map2_pyr1_warped), dim=1)

        if not self.training: # timer in inference
            warpTimeBlock2 = (time.time() - timer_p2)*1000.0
            warp_time = warp_time + warpTimeBlock2
            timer_p2 = time.time()
        
        conv_out_3 = self.convs_block_3(cat_feature_maps_pyr1) # torch.Size([batch_size, 3]) for TransNet
        batch_nn_trans_3 = self.fc_block_3(conv_out_3.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_3

        if self.show_img and self.num_blocks == 3:
            self.show_imgs_after(batch_img1, batch_img2_deRot, batch_nn_trans_sum, zeroRotMtrx, batch_planeNormVecImg1)

        if self.training:
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_3, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_4,valid_pixel_mask = self.img_warper_dict[0].transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_4 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_4*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)
            if self.self_sup:
                batch_nn_blocks_list.append(batch_homo8)
            return batch_nn_blocks_list
        else:
            netTimeBlock3 = (time.time() - timer_p3)*1000.0
            net_time = net_time + netTimeBlock3
            # timer_p4 = time.time()

            if self.trace_model: # for cpp
                batch_homo8 = compose_trans_single(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                return batch_homo8 
            else:
                batch_homo8 = compose_trans(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                return batch_homo8, warp_time, net_time


def PoseNet_ICSTN_FPE(list_blocks_types, img_height, img_width, device, self_supervised, pretrained_model=None, normalization=None, 
                      share_fc_weights=False, share_conv_weights=False, fc_dropout_rate=0.0, show_img=False, trace_model=False):

    PoseNet_FPE_model = FPE_Net(list_blocks_types, img_height, img_width, device, self_supervised, normalization=normalization, 
                                share_fc_weights=share_fc_weights, fc_dropout_rate=fc_dropout_rate, show_img=show_img, trace_model=trace_model)
                                

    if pretrained_model is not None: # use pre-trained model
        # print(data['state_dict'])
        PoseNet_FPE_model.load_state_dict(pretrained_model['state_dict'])
        print("Loaded the pre-trained Model!")

    total_params_count = sum(p.numel() for name, p in PoseNet_FPE_model.named_parameters() if p.requires_grad)
    print("Total number of trainable parameters in {} PoseNet blocks: {}".format(len(list_blocks_types), total_params_count))
    return PoseNet_FPE_model