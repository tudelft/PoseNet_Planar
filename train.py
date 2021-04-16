import torch
import torch.utils.data
# from torchsummary import summary
import time
import math

from util import charbonnier, absolutePose2homo8Pose, AverageMeter
from params import args, device


def train_pose(train_loader, PoseNet_model, optimizer, epoch, train_writer, n_iter):

    global args

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_endPoint_loss = AverageMeter()

    # ssim_loss = pytorch_ssim.SSIM() # TODO

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    # switch to train mode
    PoseNet_model.train()

    end = time.time()

    # go through the training set for one epoch and train 
    for batch_index, (inputImgs, targetPoses) in enumerate(train_loader):

        n_iter += 1
        if batch_index >= epoch_size:
            break

        batch_size = targetPoses.size()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        absolute_poseGT = targetPoses.to(device) 

        # # NOTE this is the pose at the start time point of exposure of an image
        # pose_img1 = absolute_poseGT[:, 0, :] # torch.Size([batch_size, 6])
        # pose_img2 = absolute_poseGT[:, 2, :]
        # NOTE can be the average value of the start and end point of exposure.
        pose_img1 = (absolute_poseGT[:, 0, :] + absolute_poseGT[:, 1, :]) / 2; # torch.Size([batch_size, 6])
        pose_img2 = (absolute_poseGT[:, 2, :] + absolute_poseGT[:, 3, :]) / 2; 
                
        input_img1 = inputImgs[0].to(device)
        input_img2 = inputImgs[1].to(device) # torch.Size([batch_size, 3, 320, 448])

        loss_block_sum = torch.zeros(1).squeeze().to(device)

        # 1. ground truth # torch.Size([batch_size, 8])
        homo8_1to2_GT,homo8_1to2_initial = absolutePose2homo8Pose(pose_img1, pose_img2, batch_size, rot_random_bias=0.0, slope_random_bias=0.0) # NOTE train without attitude noise
        homo8_2to1_GT,homo8_2to1_initial = absolutePose2homo8Pose(pose_img2, pose_img1, batch_size, rot_random_bias=0.0, slope_random_bias=0.0)

        # 2. nn forward
        # torch.autograd.set_detect_anomaly(True)
        homo8_1to2_nn_blockList = PoseNet_model(input_img1, input_img2, homo8_1to2_initial)
        homo8_2to1_nn_blockList = PoseNet_model(input_img2, input_img1, homo8_2to1_initial)

        # 3. loss
        for block_num in range(PoseNet_model.num_blocks):

            if args.self_supervised: # self-supervised
		# charbonnier loss
                block_loss = torch.mean(charbonnier(homo8_1to2_nn_blockList[block_num])) + torch.mean(charbonnier(homo8_2to1_nn_blockList[block_num]))

                # # L1 loss
                # block_loss = torch.mean(torch.nn.L1Loss()(homo8_1to2_nn_blockList[block_num], torch.zeros(homo8_1to2_nn_blockList[block_num].size()).to(device))) + \
                #              torch.mean(torch.nn.L1Loss()(homo8_2to1_nn_blockList[block_num], torch.zeros(homo8_2to1_nn_blockList[block_num].size()).to(device)))

                # TODO SSIM ssim_loss
            else: # supervised with ground truth
                homo8_1to2_blockError = homo8_1to2_GT - homo8_1to2_nn_blockList[block_num]
                homo8_2to1_blockError = homo8_2to1_GT - homo8_2to1_nn_blockList[block_num]
                block_loss = torch.mean(charbonnier(homo8_1to2_blockError[:, 5:])) + torch.mean(charbonnier(homo8_2to1_blockError[:, 5:])) # NOTE only trans error for now
            
            # predict tilt TODO
            # block_loss = torch.mean(charbonnier(homo8_1to2_blockError[:, 0:2])) + torch.mean(charbonnier(homo8_2to1_blockError[:, 0:2]))

            loss_block_sum = loss_block_sum + block_loss * args.list_blocks_weights[block_num] 
            # this creates a new tensor in a new address with the same name. # print(id(loss_block_sum))

        loss = loss_block_sum

        # compute gradient and do optimization step # torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure time consuming
        batch_time.update(time.time() - end)
        end = time.time()

        # in self-supervised training, the block_loss is the photometric error, 
        # while validate_endPoint_loss is the pose error, not comparable! So here recalculate the pose error
        if args.self_supervised:
            homo8_1to2_blockError = homo8_1to2_GT - homo8_1to2_nn_blockList[PoseNet_model.num_blocks]
            homo8_2to1_blockError = homo8_2to1_GT - homo8_2to1_nn_blockList[PoseNet_model.num_blocks]
            block_loss = torch.mean(charbonnier(homo8_1to2_blockError[:, 5:])) + torch.mean(charbonnier(homo8_2to1_blockError[:, 5:])) # NOTE only trans error for now

        train_endPoint_loss.update(block_loss.item(), n=batch_size) # the final loss is the last block_loss

        if batch_index % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Bidirection_Loss {5}'
                .format(epoch, batch_index, epoch_size, batch_time,
                        data_time, train_endPoint_loss))
        if args.save_train_log:
            train_writer.add_scalar('train_loss_bidirection', loss.item(), n_iter)

    print('   EPE loss train: {:.3f}, Epoch: {}'.format(train_endPoint_loss.avg, epoch))
    return train_endPoint_loss.avg, n_iter


def validate_pose(validate_loader, PoseNet_model, epoch, validate_writer): 

    global args

    batch_time = AverageMeter()

    warping_time = AverageMeter()
    net_time = AverageMeter()

    validate_endPoint_loss = AverageMeter()

    epoch_size = len(validate_loader) if args.epoch_size == 0 else min(len(validate_loader), args.epoch_size)

    # switch to evaluate mode
    PoseNet_model.eval() 

    end = time.time()
    
    for batch_index, (inputImgs, targetPoses) in enumerate(validate_loader): # 

        if batch_index >= epoch_size:
            break

        batch_size = targetPoses.size()[0]

        absolute_poseGT = targetPoses.to(device) # torch.Size([8, 2, 320, 448])

        # # NOTE this is the pose at the start time point of exposure of an image
        # pose_img1 = absolute_poseGT[:, 0, :] # torch.Size([batch_size, 6])
        # pose_img2 = absolute_poseGT[:, 2, :]
        # NOTE can be the average value of the start and end point of exposure.
        pose_img1 = (absolute_poseGT[:, 0, :] + absolute_poseGT[:, 1, :]) / 2; # torch.Size([batch_size, 6])
        pose_img2 = (absolute_poseGT[:, 2, :] + absolute_poseGT[:, 3, :]) / 2; 
        
        input_img1 = inputImgs[0].to(device)
        input_img2 = inputImgs[1].to(device) # torch.Size([batch_size, 3, 320, 448])

        # 1. ground truth # torch.Size([batch_size, 8])
        homo8_1to2_GT,homo8_1to2_initial = absolutePose2homo8Pose(pose_img1, pose_img2, batch_size, rot_random_bias=0.0, slope_random_bias=0.0) # NOTE train without attitude noise
        homo8_2to1_GT,homo8_2to1_initial = absolutePose2homo8Pose(pose_img2, pose_img1, batch_size, rot_random_bias=0.0, slope_random_bias=0.0)

        # 2. nn forward
        homo8_1to2_nn, warpingTimer1, netTimer1 = PoseNet_model(input_img1,input_img2,homo8_1to2_initial)
        homo8_2to1_nn, warpingTimer2, netTimer2 = PoseNet_model(input_img2,input_img1,homo8_2to1_initial)

        if batch_index > 6: # not counting the first a few
            warping_time.update((warpingTimer1+warpingTimer2)*0.5, n=batch_size)
            net_time.update((netTimer1+netTimer2)*0.5, n=batch_size)

        # 3. error
        homo8_1to2_blockError = homo8_1to2_GT - homo8_1to2_nn
        homo8_2to1_blockError = homo8_2to1_GT - homo8_2to1_nn
        endPoint_loss = torch.mean(charbonnier(homo8_1to2_blockError[:, 5:])) + torch.mean(charbonnier(homo8_2to1_blockError[:, 5:]))
        # predict tilt TODO
        # endPoint_loss = torch.mean(charbonnier(homo8_1to2_blockError[:, 0:2])) + torch.mean(charbonnier(homo8_2to1_blockError[:, 0:2]))

        validate_endPoint_loss.update(endPoint_loss.item(), n=batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t Bidirection_Loss {3}'
                .format(batch_index, epoch_size, batch_time, validate_endPoint_loss))
    if batch_index > 6:
        print('WarpingTime(ms):', warping_time.avg, 'NetworkTime(ms):', net_time.avg, 'Total fps:', 1000.0/(warping_time.avg+net_time.avg))
    print(' * EPE loss validate: {:.3f}, Epoch: {}'.format(validate_endPoint_loss.avg, epoch))
    return validate_endPoint_loss.avg 
