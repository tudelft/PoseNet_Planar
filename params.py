import argparse
import torch
import os

import models
import datasets

dataset_names = sorted(name for name in datasets.__all__)
model_names = sorted(name for name in models.__all__) # 

parser = argparse.ArgumentParser(description='PyTorch PoseNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# dataset
parser.add_argument('--dataset', metavar='DATASET', default='mscoco_planar',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
# parser.add_argument('--grayscale', default=True, type=bool,
#                     help='whether the input images to the network are gray-scale')
parser.add_argument('--img-height', default=224, type=float,
                    help='image height')
parser.add_argument('--img-width', default=320, type=float,
                    help='image width')
parser.add_argument('--dataset-dir', default='example_images/', type=str,
                    help='the path to the dataset')

# network architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='PoseNet_ICSTN_Standard', #
                    choices=model_names,
                    help='network architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--number-block', default=3, type=int, metavar='N',
                    help='how many network blocks in the ICSTN framework')
# parser.add_argument('--list-blocks-types', default=["trans"], type=str,
#                     help='list-pose-pred-blocks') # NOTE for now only have trans. too long, will be set below
# parser.add_argument('--list-blocks-weights', default=[1.0], type=str,
#                     help='list-pose-pred-blocks') # NOTE too long, will be set below
# fully-connected layer
# parser.add_argument('--fc', default=True, type=bool,
#                     help='last layer to be fully connected layer. If false, kernel_size=1 and average pooling.')
parser.add_argument('--fc-dropout-rate', default=0.0, type=float,
                    help='')
# weight sharing
parser.add_argument('--share-fc', action='store_true',
                    help='share the weights of the last fully-connected layer among blocks. Only applies to ICSTN_Standard and ICSTN_FPE that all the blocks have the identical fc layer arch.')
parser.add_argument('--share-conv', action='store_true',
                    help='only applies to ICSTN Standard that all the blocks have the identical arch')

# training
parser.set_defaults(self_supervised=True)
# parser.add_argument('--self-supervised', default=True, type=bool,
#                     help='train the network using photometric error loss')
parser.add_argument('--supervised', dest='self_supervised', action='store_false',
                    help='train the network using ground truth')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--learning-rate', default=2.0e-4, type=float,
                    metavar='LRP', help='initial learning rate of pose model')
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, # 
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', '--bd', default=0.0, type=float, #
                    metavar='B', help='bias decay')
parser.add_argument('--normalization', default='batch', type=str,
                    help='batch or group normalization') # NOTE no normalization is utilized for now
parser.add_argument('--use-pretrained', action='store_true',
                    help='whether to use a pre-trained model of PoseNet')
parser.add_argument('--model-path', default='checkpoint.pth.tar', type=str,
                    help='the path to the pre-trained model') #
# epoch
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', default=[5, 10, 15, 20], metavar='N', nargs='*', 
                    help='epochs at which learning rate is divided by 2')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', # for very big dataset that we do not want to train with all the samples in one epoch
                    help='manual epoch size (will match dataset size if set to 0)')
# device
parser.add_argument('--gpu', default='0', type=str,
                    help='the number of GPU to use') #

# output for user
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--show-img', action='store_true', 
                    help='show the images in network training and inference, functions in models/**.py')
parser.add_argument('--save-train-log', default=True, type=bool,
                    help='whether save trainning logging data for tensorboard curves')
parser.add_argument('--save-model', default=True, type=bool,
                    help='whether save model(trained) during training')
# inference
parser.add_argument('--test', action='store_true',
                    help='evaluate pose model on test set')
parser.add_argument('--flight', action='store_true',
                    help='run the trained network model on UZH-FPV dataset')
# for C++
parser.add_argument('--trace-model', action='store_true', # for save the trained network as a file that can be loaded in C++
                    help='TorchScript')

global args
args = parser.parse_args()

# key params
args.arch = 'PoseNet_ICSTN_Pyramid' # PoseNet_ICSTN_Standard   PoseNet_ICSTN_Pyramid   PoseNet_ICSTN_FPE
args.number_block = 4
# args.self_supervised = False
# args.batch_size = 16
# args.learning_rate = 2.0e-4 # default: 2.0e-4
# args.milestones = [5, 10, 15, 20]
# args.print_freq = 2

args.dataset_dir = 'example_images/' # NOTE here is an example. Change it to your own here.
args.train_folder = 'train' 
args.validate_folder = 'val'
args.test_folder = 'val'

if args.number_block == 1:
     args.list_blocks_types = ["trans"]
     args.list_blocks_weights = [1.0]
elif args.number_block == 2:
     args.list_blocks_types = ["trans", "trans"]
     args.list_blocks_weights = [0.3, 0.7]
elif args.number_block == 3:
     args.list_blocks_types = ["trans", "trans", "trans"]
     args.list_blocks_weights = [0.2, 0.3, 0.5]     
elif args.number_block == 4:
     args.list_blocks_types = ["trans", "trans", "trans", "trans"]
     args.list_blocks_weights = [0.1, 0.2, 0.3, 0.4]

# pretrained models
if args.number_block == 1 and args.arch == 'PoseNet_ICSTN_Standard':
     args.model_path = 'pretrained_models/TableI-2th.pth.tar' #
if args.number_block == 3 and args.arch == 'PoseNet_ICSTN_Standard':
     args.model_path = 'pretrained_models/TableI-6th.pth.tar' #
if args.number_block == 3 and args.arch == 'PoseNet_ICSTN_FPE': # for now FPE net only supports 3 blocks
     args.model_path = 'pretrained_models/TableII-6th.pth.tar' #
if args.number_block == 4 and args.arch == 'PoseNet_ICSTN_Pyramid':
     args.model_path = 'pretrained_models/TableIII-4th.pth.tar' #

if torch.cuda.is_available():
     device = torch.device("cuda:"+args.gpu)
     print("GPU available! Using", device)
else:
     device = torch.device("cpu")
     print("CPU Debugging Mode! Using", device)

if args.test:
     args.save_train_log = False

if args.save_train_log:
     print("Saving Training Log!")
else:
     print("WARNING! NOT Saving Training Log!")

if args.save_model:
     print("Saving Best Model Params!")
else:
     print("WARNING! NOT Saving Model Params!")

if args.self_supervised:
     print("Self-Supervised Learning!")
else:
     print("Supervised Learning with Ground Truth!")
