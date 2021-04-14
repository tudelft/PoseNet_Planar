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
#     'PoseNet_ICSTN_Pyramid'
# ]

def conv(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
        nn.ReLU(inplace=True)
    )

class ICSTNPyramid(nn.Module): # build Inverse Compositional STN

    def __init__(self, list_blocks_types, img_height, img_width, device, self_supervised, normalization=None, 
                 share_fc_weights=True, fc_dropout_rate=0.0, show_img=False, trace_model = False):

        super(ICSTNPyramid,self).__init__()

        self.device = device
        self.max_pyramid_level = len(list_blocks_types) - 1 # each block corresponds to one pyramid level
        self.num_blocks = len(list_blocks_types)
        self.show_img = show_img 
        self.share_fc = share_fc_weights
        self.trace_model = trace_model
        self.self_sup = self_supervised
        # self.self_sup_SSIM = False # TODO

        self.img_warper = warp.WarpImg(img_height=img_height, img_width=img_width, device=self.device)

        self.dof = [] 
        for motion in list_blocks_types:
            if motion == 'trans' or motion == 'rot': 
                self.dof.append(3) # NOTE degree-of-freedom to predict (for now translation only)
            elif motion == 'tilt':
                self.dof.append(2)     
            elif motion == 'rot&trans':
                self.dof.append(6)

        self.avgPool = False # True False

        if self.num_blocks == 4:
            conv_layers_block_list = [2.7522, 2.7542, 3.7542, 3.7544] # TABLE II 4th
            linear_inputs = [8960, 8960, 17920, 17920] # NOTE cannot share weights among fully-connected layers
        elif self.num_blocks == 3:
            conv_layers_block_list = [4.7522,4.7542,4.7544] # TABLE II 3rd
            linear_inputs = [5120, 5120, 5120]
        else:
            print("Error! No pyramidal network block is implemented for", list_blocks_types)

        assert(self.num_blocks == len(conv_layers_block_list))

        if self.share_fc:
            if len(set(linear_inputs)) != 1 or len(set(self.dof)) != 1:
                self.share_fc = False
                print("Cannot share fully-connected layer with this architecture! Will use different weights.")

        # create network blocks

        self.convs_block_1 = self.create_convs(conv_layers_block_list[0])
        self.linear_input_1 = linear_inputs[0]
        self.fc_block_1 = nn.Linear(self.linear_input_1, self.dof[0], bias=True)

        if self.num_blocks > 1:
            self.convs_block_2 = self.create_convs(conv_layers_block_list[1])

            if self.share_fc:
                assert(linear_inputs[1] == linear_inputs[0])
                self.fc_block_2 = self.fc_block_1
            else:
                self.linear_input_2 = linear_inputs[1]
                self.fc_block_2 = nn.Linear(self.linear_input_2, self.dof[1], bias=True)

        if self.num_blocks > 2:
            self.convs_block_3 = self.create_convs(conv_layers_block_list[2])

            if self.share_fc:
                assert(linear_inputs[2] == linear_inputs[0])
                self.fc_block_3 = self.fc_block_1
            else:
                self.linear_input_3 = linear_inputs[2]
                self.fc_block_3 = nn.Linear(self.linear_input_3, self.dof[2], bias=True)

        if self.num_blocks > 3:
            self.convs_block_4 = self.create_convs(conv_layers_block_list[3])
        
            if self.share_fc:
                assert(linear_inputs[3] == linear_inputs[0])
                self.fc_block_4 = self.fc_block_1
            else:
                self.linear_input_4 = linear_inputs[3]
                self.fc_block_4 = nn.Linear(self.linear_input_4, self.dof[3], bias=True)

        print("Pyramidal PoseNet has {} image pyramid levels.".format(self.max_pyramid_level+1))
        print("PoseNet blocks are:", list_blocks_types, conv_layers_block_list)
        # print("Input fully connected:", linear_inputs)

        if self.avgPool:
            print("Average Pooling for downsampling.")
        else:
            print("Bilinear Interpolation for downsampling.")

        self.init_weights()
        print("ICSTN Pyramid is initialized!")
        
        if self.show_img:
            cv2.namedWindow('before', cv2.WINDOW_NORMAL)
            cv2.namedWindow('after', cv2.WINDOW_NORMAL)

    def init_weights(self):
        print("Conv weights xavier_uniform_ !")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight) # how about kaiming_normal_ ? # kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu') #
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # zeros_(m.weight) # use pytorch default init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if m.bias is not None:
                    zeros_(m.bias)

    def create_convs(self, layers):

        if layers == 2.7522:
            conv_planes = [64, 128]
            posenet_block = nn.Sequential( # original resolution 224 320, input resolution 28 40
                conv(             2, conv_planes[0], kernel_size=7, stride=2),
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2) 
                ) # Receptive Field = 15*8
            if self.avgPool:
                posenet_block = nn.Sequential(nn.AvgPool2d(8, stride=8, padding=0),posenet_block)
            return posenet_block
        
        if layers == 2.7542:
            conv_planes = [64, 128]
            posenet_block = nn.Sequential( # original resolution 224 320, input resolution 56 80
                conv(             2, conv_planes[0], kernel_size=7, stride=4), 
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2)
                ) # Receptive Field = 23*4
            if self.avgPool:
                posenet_block = nn.Sequential(nn.AvgPool2d(4, stride=4, padding=0),posenet_block)
            return posenet_block
        
        if layers == 3.7542:
            conv_planes = [32, 128, 256] 
            posenet_block = nn.Sequential( # 112 160
                conv(             2, conv_planes[0], kernel_size=7, stride=4), # 28 40
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2), # 14 20
                conv(conv_planes[1], conv_planes[2], stride=2)  #  7 10
                )# Receptive Field = 39*2
            if self.avgPool:
                posenet_block = nn.Sequential(nn.AvgPool2d(2, stride=2, padding=0),posenet_block)
            return posenet_block 

        if layers == 3.7544: 
            conv_planes = [32, 128, 256] 
            posenet_block = nn.Sequential( # 224 320
                conv(             2, conv_planes[0], kernel_size=7, stride=4), # 56 80
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=4), # 14 20
                conv(conv_planes[1], conv_planes[2], stride=2)  # 7 10
                ) # Receptive Field = 55
            return posenet_block 
        
        if layers == 4.7522:
            conv_planes = [32, 64, 128, 256]
            posenet_block = nn.Sequential( # 56 80
                conv(             2, conv_planes[0], kernel_size=7, stride=2), # 28 40
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2), # 14 20
                conv(conv_planes[1], conv_planes[2], stride=2), # 7 10
                conv(conv_planes[2], conv_planes[3], stride=2)  # 4 5 
                ) # Receptive Field = 39
            if self.avgPool:
                posenet_block = nn.Sequential(nn.AvgPool2d(4, stride=4, padding=0),posenet_block)
            return posenet_block

        if layers == 4.7542:
            conv_planes = [32, 64, 128, 256]
            posenet_block = nn.Sequential( # 112 160
                conv(             2, conv_planes[0], kernel_size=7, stride=4), # 28 40
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2), # 14 20
                conv(conv_planes[1], conv_planes[2], stride=2), # 7 10
                conv(conv_planes[2], conv_planes[3], stride=2) # 4 5
                ) # Receptive Field = 71
            if self.avgPool:
                posenet_block = nn.Sequential(nn.AvgPool2d(2, stride=2, padding=0),posenet_block)
            return posenet_block

        if layers == 4.7544:
            conv_planes = [32, 64, 128, 256]
            posenet_block = nn.Sequential( # 224 320
                conv(             2, conv_planes[0], kernel_size=7, stride=4), # 56 80
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=4), # 14 20
                conv(conv_planes[1], conv_planes[2], stride=2), # 7 10
                conv(conv_planes[2], conv_planes[3], stride=2) # 4 5
                ) # Receptive Field = 119
            return posenet_block

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    
    def show_imgs_after(self, batch_img1, batch_img2, batch_trans, batch_rotMtrx, batch_planeNormVecImg1):
        batch_img2_warped = self.img_warper.transformImage(batch_img2,batch_trans,batch_rotMtrx,batch_planeNormVecImg1)
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
        
        if self.trace_model: # for cpp
            batch_img2_deRot = self.img_warper.transformImage_Single(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1)
        else:
            batch_img2_deRot = self.img_warper.transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1)

        if self.show_img:
            imgs_before = torch.abs(batch_img1 - batch_img2_deRot)[0, 0, :, :]
            show_imgs = torch.cat((batch_img1[0, :, :, :], batch_img2_deRot[0, :, :, :], imgs_before.unsqueeze(0)), dim=1).to('cpu')
            cv2.imshow('before', show_imgs.detach().numpy().transpose(1, 2, 0))
            cv2.waitKey(-1) 

        ## block 1 (smallest resolution)
        cat_imgs = torch.cat((batch_img1, batch_img2_deRot), dim=1)

        if self.avgPool:
            cat_imgs_downsampled = cat_imgs
        else:
            cat_imgs_downsampled = downsampling(self.max_pyramid_level, cat_imgs)

        if not self.training: # timer in inference
            deRotTime = (time.time() - timer_p0)*1000.0
            warp_time = warp_time + deRotTime
            timer_p1 = time.time()

        conv_out_1 = self.convs_block_1(cat_imgs_downsampled)
        batch_nn_trans_1 = self.fc_block_1(conv_out_1.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_1

        if self.training:
            batch_nn_blocks_list = []
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_1, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_2,valid_pixel_mask = self.img_warper.transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
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
            batch_img2_warped_2 = self.img_warper.transformImage_Single(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        else:
            batch_img2_warped_2 = self.img_warper.transformImage(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)

        cat_imgs = torch.cat((batch_img1, batch_img2_warped_2), dim=1) # torch.Size([batch_size, 2, H, W])
        if self.max_pyramid_level == 1 or self.avgPool:
            cat_imgs_downsampled = cat_imgs
        else:
            cat_imgs_downsampled = downsampling(self.max_pyramid_level-1, cat_imgs)
            
        if not self.training: # timer in inference
            warpTimeBlock2 = (time.time() - timer_p2)*1000.0
            warp_time = warp_time + warpTimeBlock2
            timer_p2 = time.time()
        
        conv_out_2 = self.convs_block_2(cat_imgs_downsampled)
        batch_nn_trans_2 = self.fc_block_2(conv_out_2.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_2

        if self.show_img and self.num_blocks == 2:
            self.show_imgs_after(batch_img1, batch_img2_deRot, batch_nn_trans_sum, zeroRotMtrx, batch_planeNormVecImg1)

        if self.training:
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_2, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_3,valid_pixel_mask = self.img_warper.transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_3 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_3*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)
            if self.num_blocks == 2:
                if self.self_sup:
                    batch_nn_blocks_list.append(batch_homo8)
                return batch_nn_blocks_list
        else:
            netTimeBlock2 = (time.time() - timer_p2)*1000.0
            net_time = net_time + netTimeBlock2
            timer_p3 = time.time()

            if self.num_blocks == 2:
                if self.trace_model: # for cpp
                    batch_homo8 = compose_trans_single(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8 
                else:
                    batch_homo8 = compose_trans(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8, warp_time, net_time 

        ## block 3
        if self.trace_model: # for cpp
            batch_img2_warped_3 = self.img_warper.transformImage_Single(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        else:
            batch_img2_warped_3 = self.img_warper.transformImage(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)

        cat_imgs = torch.cat((batch_img1, batch_img2_warped_3), dim=1)
        if self.max_pyramid_level == 2 or self.avgPool:
            cat_imgs_downsampled = cat_imgs
        else:
            cat_imgs_downsampled = downsampling(self.max_pyramid_level-2, cat_imgs)
        
        if not self.training: # timer in inference
            warpTimeBlock3 = (time.time() - timer_p3)*1000.0
            warp_time = warp_time + warpTimeBlock3
            timer_p3 = time.time()
        
        conv_out_3 = self.convs_block_3(cat_imgs_downsampled)
        batch_nn_trans_3 = self.fc_block_3(conv_out_3.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_3

        if self.show_img and self.num_blocks == 3:
            self.show_imgs_after(batch_img1, batch_img2_deRot, batch_nn_trans_sum, zeroRotMtrx, batch_planeNormVecImg1)

        if self.training:
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_3, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_4,valid_pixel_mask = self.img_warper.transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_4 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_4*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)
            if self.num_blocks == 3:
                if self.self_sup:
                    batch_nn_blocks_list.append(batch_homo8)
                return batch_nn_blocks_list
        else:
            netTimeBlock3 = (time.time() - timer_p3)*1000.0
            net_time = net_time + netTimeBlock3
            timer_p4 = time.time()

            if self.num_blocks == 3:
                if self.trace_model: # for cpp
                    batch_homo8 = compose_trans_single(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8 
                else:
                    batch_homo8 = compose_trans(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8, warp_time, net_time 

        ## block 4
        if self.trace_model: # for cpp
            batch_img2_warped_4 = self.img_warper.transformImage_Single(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        else:
            batch_img2_warped_4 = self.img_warper.transformImage(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        
        cat_imgs = torch.cat((batch_img1, batch_img2_warped_4), dim=1) # torch.Size([batch_size, 2, 224, 320])

        if not self.training:
            warpTimeBlock4 = (time.time() - timer_p4)*1000.0
            warp_time = warp_time + warpTimeBlock4
            timer_p4 = time.time()

        conv_out_4 = self.convs_block_4(cat_imgs)
        batch_nn_trans_4 = self.fc_block_4(conv_out_4.view(batch_size, -1))
        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_4
        
        if self.show_img:
            self.show_imgs_after(batch_img1, batch_img2_deRot, batch_nn_trans_sum, zeroRotMtrx, batch_planeNormVecImg1)

        if self.training:
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_4, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_5,valid_pixel_mask = self.img_warper.transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_5 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_5*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)
            if self.num_blocks == 4:
                if self.self_sup:
                    batch_nn_blocks_list.append(batch_homo8)
                return batch_nn_blocks_list
        else:
            netTimeBlock4 = (time.time() - timer_p4)*1000.0
            net_time = net_time + netTimeBlock4
            # timer_p5 = time.time()

            if self.num_blocks == 4:
                if self.trace_model: # for cpp
                    batch_homo8 = compose_trans_single(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8 
                else:
                    batch_homo8 = compose_trans(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8, warp_time, net_time 

        
def PoseNet_ICSTN_Pyramid(list_blocks_types, img_height, img_width, device, self_supervised, pretrained_model=None, normalization=None, 
                          share_fc_weights=False, share_conv_weights=False, fc_dropout_rate=0.0, show_img=False, trace_model=False):

    PoseNet_ICSTN_model = ICSTNPyramid(list_blocks_types, img_height, img_width, device, self_supervised, 
                                       normalization=normalization, share_fc_weights=share_fc_weights, 
                                       fc_dropout_rate=fc_dropout_rate, show_img=show_img, trace_model=trace_model)

    if pretrained_model is not None: # use pre-trained model
        # print(data['state_dict'])
        PoseNet_ICSTN_model.load_state_dict(pretrained_model['state_dict'])
        print("Loaded the pre-trained Model!")

    total_params_count = sum(p.numel() for name, p in PoseNet_ICSTN_model.named_parameters() if p.requires_grad)
    print("Total number of trainable parameters in {} PoseNet blocks: {}".format(len(list_blocks_types), total_params_count))
    return PoseNet_ICSTN_model