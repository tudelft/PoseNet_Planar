import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import time
from torch.nn.init import kaiming_normal_, constant_, xavier_uniform_, zeros_

import warp
from util import *

# __all__ = [
#     'PoseNet_ICSTN_Standard'
# ]

def conv(in_planes, out_planes, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
        nn.ReLU(inplace=True)
    )

def convReverse(in_planes, out_planes, kernel_size=3, stride=2, LeakyReLU=False, noActi=False):
    if noActi:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
    elif LeakyReLU:
        return nn.Sequential(
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
        )
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride)
        )

# class Res2Block(nn.Module): # NOTE 2 layers of conv inside one ResBlock
#     """Residual Block keeping same channels and height and width as the input tensor https://arxiv.org/pdf/1603.05027.pdf).
#     """

#     def __init__(self, in_planes, out_planes, intermediate_planes=None, kernel_size=3, stride=1, bias=True, normalization="batch"):
#         super().__init__()

#         assert(in_planes == out_planes)
#         if intermediate_planes is None:
#             intermediate_planes = in_planes

#         self.normalization = normalization

#         #residual function
#         self.residual_function = nn.Sequential(
#             convReverse(in_planes, intermediate_planes, kernel_size=kernel_size, stride=1, LeakyReLU=True),
#             convReverse(intermediate_planes, out_planes, kernel_size=kernel_size, stride=1, LeakyReLU=True)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#     def forward(self, input):
#         return self.residual_function(input) + self.shortcut(input)

class DenseBlock(nn.Module): # NOTE 2 layers of conv inside one DenceBlock
    """DenceNet Block keeping same height and width as the input tensor 
    https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
    """
    def __init__(self, in_planes, out_planes, intermediate_planes=None, kernel_size=3, stride=2, bias=True, normalization="batch"):
        super().__init__()

        if intermediate_planes is None:
            intermediate_planes = in_planes

        self.normalization = normalization 
        
        self.net1 = convReverse(in_planes, intermediate_planes, kernel_size=kernel_size, stride=1, LeakyReLU=True)
        self.net2 = convReverse(intermediate_planes+in_planes, intermediate_planes, kernel_size=kernel_size, stride=1, LeakyReLU=True)
        self.net3 = convReverse(intermediate_planes+intermediate_planes+in_planes, out_planes, kernel_size=kernel_size, stride=stride, LeakyReLU=True)

    def forward(self, input):

        tensor_1 = self.net1(input)
        tensor_2 = self.net2(torch.cat((input, tensor_1),dim=1))
        tensor_3 = self.net3(torch.cat((input, tensor_1, tensor_2),dim=1))

        return tensor_3


# build Inverse Compositional STN
class ICSTNStandard(nn.Module):

    def __init__(self, list_blocks_types, img_height, img_width, device, self_supervised, normalization=None, 
                 share_fc_weights=False, share_conv_weights=False, fc_dropout_rate=0.0, show_img=False, trace_model = False):

        super(ICSTNStandard,self).__init__()

        self.device = device
        self.num_blocks = len(list_blocks_types)
        self.show_img = show_img 
        self.share_fc = share_fc_weights
        self.share_conv = share_conv_weights
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

        if self.share_fc:
            if len(set(self.dof)) != 1:
                self.share_fc = False
                print("Cannot share fully-connected layer with this architecture! Will use different weights.")

        if self.num_blocks == 1:
            conv_layers_block = 18 # TABLE I 2nd
            linear_inputs = 768
        elif self.num_blocks == 3:
            conv_layers_block = 5.7542 # TABLE I 6th
            linear_inputs = 5120
        else:
            print("Error! No ICSTN network block is implemented for", list_blocks_types)

        # create network blocks

        # block 1
        self.convs_block_1 = self.create_convs(conv_layers_block)
        self.fc_block_1 = nn.Linear(linear_inputs, self.dof[0], bias=True)

        # block 2
        if self.num_blocks > 1:

            if self.share_conv:
                self.convs_block_2 = self.convs_block_1
            else:
                self.convs_block_2 = self.create_convs(conv_layers_block)

            if self.share_fc:
                assert(self.dof[0] == self.dof[1])
                self.fc_block_2 = self.fc_block_1
            else:
                self.fc_block_2 = nn.Linear(linear_inputs, self.dof[1], bias=True)

        # block 3
        if self.num_blocks > 2:

            if self.share_conv:
                self.convs_block_3 = self.convs_block_1
            else:
                self.convs_block_3 = self.create_convs(conv_layers_block)

            if self.share_fc:
                assert(self.dof[0] == self.dof[2])
                self.fc_block_3 = self.fc_block_1
            else:
                self.fc_block_3 = nn.Linear(linear_inputs, self.dof[2], bias=True)

        print("PoseNet blocks are:", list_blocks_types, conv_layers_block)

        self.init_weights()
        print("ICSTN Standard is initialized!")
        
        if self.show_img:
            cv2.namedWindow('before', cv2.WINDOW_NORMAL)
            cv2.namedWindow('after', cv2.WINDOW_NORMAL)

    def init_weights(self):
        print("Conv weights xavier_uniform_ !")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # zeros_(m.weight) # use pytorch default init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if m.bias is not None:
                    zeros_(m.bias)

    def create_convs(self, layers):

        if layers == 5.7542:

            if self.share_conv:
                conv_planes = [64, 128, 128, 256, 256]
            else:
                conv_planes = [16, 32, 64, 128, 256]

            posenet_block = nn.Sequential( # 224 320
                conv(             2, conv_planes[0], kernel_size=7, stride=4), # 56 80
                conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2), # 28 40
                conv(conv_planes[1], conv_planes[2], stride=2), # 14 20
                conv(conv_planes[2], conv_planes[3], stride=2), #  7 10
                conv(conv_planes[3], conv_planes[4], stride=2)  #  4  5  # Receptive Field 135
                )
            return posenet_block

        if layers == 18:  # densenet
            conv_planes = [16, 32, 32, 32, 48, 48, 48, 64, 64, 64, 80, 80, 80, 96, 96, 96, 128] # 17 numbers here, [1] is used twice

            posenet_block = nn.Sequential( # 224 320  
                convReverse(             2, conv_planes[0], stride=2, noActi=True), # 112 160
                convReverse(conv_planes[0], conv_planes[1], stride=2, LeakyReLU=True), # 56 80
                convReverse(conv_planes[1], conv_planes[1], stride=1, LeakyReLU=True), # 56 80
                DenseBlock( conv_planes[1], conv_planes[4], stride=2), # 28 40
                DenseBlock( conv_planes[4], conv_planes[7], stride=2), # 14 20
                DenseBlock( conv_planes[7], conv_planes[10], stride=2), #  7 10
                DenseBlock(conv_planes[10], conv_planes[13], stride=2), # 4  5
                DenseBlock(conv_planes[13], conv_planes[16], stride=2), # 2  3
                nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )

            return posenet_block


    def weight_parameters(self):
        # weight_param_names = [name for name, param in self.named_parameters()]
        # print("ICSTN weight params: ", weight_param_names)
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

        ## block 1 
        cat_imgs = torch.cat((batch_img1, batch_img2_deRot), dim=1) # torch.Size([batch_size, 2, 224, 320])

        if not self.training: # timer in inference
            deRotTime = (time.time() - timer_p0)*1000.0
            warp_time = warp_time + deRotTime
            timer_p1 = time.time()

        conv_out_1 = self.convs_block_1(cat_imgs)
        batch_nn_trans_1 = self.fc_block_1(conv_out_1.view(batch_size, -1))

        # TODO tile prediction
        # batch_homo8_tilt_updated = batch_homo8.clone()
        # batch_homo8_tilt_updated[:, 0:2] = batch_nn_1

        batch_nn_trans_sum = batch_nn_trans_sum + batch_nn_trans_1

        if self.show_img and self.num_blocks == 1:
            self.show_imgs_after(batch_img1, batch_img2_deRot, batch_nn_trans_sum, zeroRotMtrx, batch_planeNormVecImg1)

        if self.training:
            batch_nn_blocks_list = []
            batch_homo8 = compose_trans(batch_size, batch_nn_trans_1, batch_homo8, batch_rotMtrx) # 
            if self.self_sup:
                batch_img2_warped_2,valid_pixel_mask = self.img_warper.transformImage(batch_img2,batch_homo8[:, 5:],batch_rotMtrx,batch_planeNormVecImg1,move_out_mask=True)
                batch_nn_blocks_list.append((batch_img2_warped_2 - batch_img1)*valid_pixel_mask.float())
                # batch_nn_blocks_list.append([batch_img2_warped_2*valid_pixel_mask.float(), batch_img1*valid_pixel_mask.float()]) # TODO SSIM
            else:
                batch_nn_blocks_list.append(batch_homo8)
            if self.num_blocks == 1:
                if self.self_sup:
                    batch_nn_blocks_list.append(batch_homo8)
                return batch_nn_blocks_list
        else:
            netTimeBlock1 = (time.time() - timer_p1)*1000.0
            net_time = net_time + netTimeBlock1
            timer_p2 = time.time()

            if self.num_blocks == 1:
                if self.trace_model: # for cpp
                    batch_homo8 = compose_trans_single(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8 
                else:
                    batch_homo8 = compose_trans(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8, warp_time, net_time 

        ## block 2
        if self.trace_model: # for cpp
            batch_img2_warped_2 = self.img_warper.transformImage_Single(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)
        else:
            batch_img2_warped_2 = self.img_warper.transformImage(batch_img2_deRot,batch_nn_trans_sum,zeroRotMtrx,batch_planeNormVecImg1)

        cat_imgs = torch.cat((batch_img1, batch_img2_warped_2), dim=1) # torch.Size([batch_size, 2, H, W])
        
        if not self.training: # timer in inference
            warpTimeBlock2 = (time.time() - timer_p2)*1000.0
            warp_time = warp_time + warpTimeBlock2
            timer_p2 = time.time()
        
        conv_out_2 = self.convs_block_2(cat_imgs)
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
        
        if not self.training: # timer in inference
            warpTimeBlock3 = (time.time() - timer_p3)*1000.0
            warp_time = warp_time + warpTimeBlock3
            timer_p3 = time.time()

        conv_out_3 = self.convs_block_3(cat_imgs)
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
            # timer_p4 = time.time()

            if self.num_blocks == 3:
                if self.trace_model: # for cpp
                    batch_homo8 = compose_trans_single(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8 
                else:
                    batch_homo8 = compose_trans(batch_size, batch_nn_trans_sum, batch_homo8, batch_rotMtrx)
                    return batch_homo8, warp_time, net_time 


def PoseNet_ICSTN_Standard(list_blocks_types, img_height, img_width, device, self_supervised, pretrained_model=None, normalization=None, 
                           share_fc_weights=False, share_conv_weights=False, fc_dropout_rate=0.0, show_img=False, trace_model=False):

    PoseNet_ICSTN_model = ICSTNStandard(list_blocks_types, img_height, img_width, device, self_supervised, 
                                       normalization=normalization, share_fc_weights=share_fc_weights, share_conv_weights=share_conv_weights,
                                       fc_dropout_rate=fc_dropout_rate, show_img=show_img, trace_model=trace_model)

    if pretrained_model is not None: # use pre-trained model
        # print(data['state_dict'])
        PoseNet_ICSTN_model.load_state_dict(pretrained_model['state_dict'])
        print("Loaded the pre-trained Model!")

    total_params_count = sum(p.numel() for name, p in PoseNet_ICSTN_model.named_parameters() if p.requires_grad)
    print("Total number of trainable parameters in {} PoseNet blocks: {}".format(len(list_blocks_types), total_params_count))

    return PoseNet_ICSTN_model