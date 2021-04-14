import numpy as np
import torch
import math
import torch.nn.functional as F
import time

# from util import poseVec2homoMtrx

D2R = math.pi / 180.0

class WarpImg:

    def __init__(self, img_height, img_width, device, FoV=90*D2R): # FoV(rad) is field of view of the image in width

        self.device = device
        self.img_height = img_height
        self.img_width = img_width

        self.yaw_ground = torch.zeros(1).squeeze().to(self.device) # torch.float32
        # the yaw_old angle of the camera relative to ground plane is zero

        self.planeNormalVec = torch.FloatTensor([0.0, 0.0, 1.0]).view(3, 1).to(self.device) 
        # world frame is front-right-(gravity)down and ground plane is perpendicular/orthogonal to gravity vector

        fx = (img_width-1)/2/math.tan(FoV/2)
        fy = fx
        cx = (img_width-1)/2
        cy = (img_height-1)/2
        # NOTE TODO for now only support images with "standard" intrinsic (field of view = 90 deg in width)
        camMtrx = \
            [[1*fx,     0,      1*cx],
             [0,        1*fy,   1*cy],
             [0,        0,         1]]

        camMtrx_inverse = \
            [[ 1/fx,    0, -cx/fx],
             [    0, 1/fy, -cy/fy],
             [    0,    0,      1]]
        
        camMtrx_np = np.array(camMtrx, dtype=np.float32) # list to numpy array
        self.camMtrx = torch.from_numpy(camMtrx_np).to(self.device) # numpy array to tensor

        camMtrx_inverse_np = np.array(camMtrx_inverse, dtype=np.float32) # list to numpy array
        self.camMtrx_inverse = torch.from_numpy(camMtrx_inverse_np).to(self.device) # numpy array to tensor
        
        self.grid_xy1 = self.generate_grid().to(self.device)  # torch.float32 [3, height*width]

        self.sample_grid_factor = torch.FloatTensor([[[2/(img_width-1), 2/(img_height-1)]]]).to(self.device) # torch.Size([1, 1, 2])

        print("Image Warper for image size {} {} is initialized!".format(str(img_height), str(img_width)))


    def generate_grid(self):

        W = self.img_width
        H = self.img_height

        u = torch.arange(0, W).view(1, -1).repeat(H, 1).unsqueeze(0).float().to(self.device) 
        v = torch.arange(0, H).view(-1, 1).repeat(1, W).unsqueeze(0).float().to(self.device)
        self.grid_uv = torch.cat((u, v), dim=0)

        grid_uv1 = torch.cat((self.grid_uv, torch.ones([1, H, W]).to(self.device)), dim=0) 
        grid_xy1 = torch.mm(self.camMtrx_inverse, grid_uv1.view([3, H*W]))

        return grid_xy1 

    def transformImage_Single(self,batch_imgs,batch_transVec,batch_rotMtrx,batch_planeNormVec): # trace model for cpp
        
        sample_grid_normed_batch_list = []

        homoMtrx = batch_rotMtrx[0, :, :] + torch.mm(batch_transVec[0, :].view(3, 1), batch_planeNormVec[0, :].view(1, 3))

        sample_grid_xyz = torch.mm(homoMtrx, self.grid_xy1) # [3, height*width] 
        sample_grid_xy1 = sample_grid_xyz / sample_grid_xyz[2, :] 
        sample_grid_uv1 = torch.mm(self.camMtrx, sample_grid_xy1) 
        sample_grid_uv = sample_grid_uv1[0:2, :].view([2, self.img_height, self.img_width])
        sample_grid_uv = torch.transpose(sample_grid_uv, 0, 1)
        sample_grid_uv = torch.transpose(sample_grid_uv, 1, 2) # [H, W, 2]

        sample_grid_normed = sample_grid_uv * self.sample_grid_factor - 1 # [-1, 1] ([H, W, 2])

        sample_grid_normed_batch_list.append(sample_grid_normed.unsqueeze(0)) 

        batch_sample_grid_normed = torch.cat(sample_grid_normed_batch_list, dim=0)

        if torch.__version__ < '1.2.0': 
            batch_warped_imgs = F.grid_sample(batch_imgs, batch_sample_grid_normed, mode='bilinear', padding_mode='zeros')
        else: # NOTE torch 1.1.0 does not have the input parameter "align_corners", align_corners=True by default
            batch_warped_imgs = F.grid_sample(batch_imgs, batch_sample_grid_normed, mode='bilinear', padding_mode='zeros', align_corners=True)

        return batch_warped_imgs

    def transformImage(self,batch_imgs,batch_transVec,batch_rotMtrx,batch_planeNormVec,move_out_mask=False): 
        
        batch_size = batch_imgs.size()[0]
        sample_grid_normed_batch_list = []

        if move_out_mask:
            move_out_mask_batch_list = []

        for i in range(batch_size):
            # end = time.time()
            homoMtrx = batch_rotMtrx[i, :, :] + torch.mm(batch_transVec[i, :].view(3, 1), batch_planeNormVec[i, :].view(1, 3))
            # print("homoMtrx", (time.time() - end)*1000.0)
            # end = time.time()

            sample_grid_xyz = torch.mm(homoMtrx, self.grid_xy1) # [3, height*width] 
            sample_grid_xy1 = sample_grid_xyz / sample_grid_xyz[2, :] 
            sample_grid_uv1 = torch.mm(self.camMtrx, sample_grid_xy1) # [3, height*width] 

            sample_grid_uv = sample_grid_uv1[0:2, :].view([2, self.img_height, self.img_width])
            sample_grid_uv = torch.transpose(sample_grid_uv, 0, 1)
            sample_grid_uv = torch.transpose(sample_grid_uv, 1, 2) # [H, W, 2]

            # print("sample_grid_uv", (time.time() - end)*1000.0)
            # end = time.time()

            sample_grid_normed = sample_grid_uv * self.sample_grid_factor - 1 # [-1, 1] ([H, W, 2])

            sample_grid_normed_batch_list.append(sample_grid_normed.unsqueeze(0)) 

            if move_out_mask:
                mask_u = torch.abs(sample_grid_normed[:, :, 0]) < 1 # [H, W]
                mask_v = torch.abs(sample_grid_normed[:, :, 1]) < 1
                mask_in = mask_u * mask_v # [H, W]
                move_out_mask_batch_list.append(mask_in.unsqueeze(0).unsqueeze(0))

        batch_sample_grid_normed = torch.cat(sample_grid_normed_batch_list, dim=0)
        if torch.__version__ < '1.2.0': 
            batch_warped_imgs = F.grid_sample(batch_imgs, batch_sample_grid_normed, mode='bilinear', padding_mode='zeros')
        else: # NOTE torch 1.1.0 does not have the input parameter "align_corners", align_corners=True by default
            batch_warped_imgs = F.grid_sample(batch_imgs, batch_sample_grid_normed, mode='bilinear', padding_mode='zeros', align_corners=True)

        if move_out_mask:
            batch_masks = torch.cat(move_out_mask_batch_list, dim=0) # ([B, 1, H, W]) # print(batch_masks.shape) torch.Size([B, 1, 224, 320])
            return batch_warped_imgs, batch_masks.detach()
        else:
            return batch_warped_imgs