from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.insert(0, "..")
import os
import random

import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from skimage import color
from torch.autograd import Variable
from utils.flowlib import read_flow
from utils.util_distortion import CenterPad
from utils.util_distortion import (CenterPad_threshold, Normalize, RGB2Lab,
                                   ToTensor)
import lib.TrainTransforms as train_transforms
from torchvision.transforms import RandomCrop

import utils as utils
import os,time,cv2,scipy.io
import scipy.misc as sic
import subprocess
import argparse
import torch
import imageio
import codecs
import torchvision.transforms as transforms


import lib.functional as F

import math

# SuperSlomo
from torchvision import transforms as superslomo_transforms
import pickle

cv2.setNumThreads(0)


class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return color.rgb2lab(inputs)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        inputs[0:1, :, :] = F.normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = F.normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        outputs = F.to_mytensor(inputs)  # permute channel and transform to tensor
        return outputs


class RandomErasing(object):
    def __init__(self, probability=0.6, sl=0.05, sh=0.6):
        self.probability = probability
        self.sl = sl
        self.sh = sh

    def __call__(self, img):
        img = np.array(img)
        if random.uniform(0, 1) > self.probability:
            return Image.fromarray(img)

        area = img.shape[0] * img.shape[1]
        h0 = img.shape[0]
        w0 = img.shape[1]
        channel = img.shape[2]

        h = int(round(random.uniform(self.sl, self.sh) * h0))
        w = int(round(random.uniform(self.sl, self.sh) * w0))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            img[x1 : x1 + h, y1 : y1 + w, :] = np.random.rand(h, w, channel) * 255
            return Image.fromarray(img)

        return Image.fromarray(img)


class CenterCrop(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy

# import from favc 
def read_image_sequence(filename, num_frames):
    filename = os.path.join(filename, 'pics')
    print(filename)
    assert 1==0
    file1 = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(os.path.basename(filename))[1]
    try:
        img1 = imageio.imread(filename).astype(np.float32)
        # img1 = sic.imread(filename).astype(np.float32)
        imgh1 = img1
    except:
        print("Cannot read the first frame.")
        return None, None
    if len(img1.shape) == 2: # discard grayscale images
        return None, None

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img1 = np.expand_dims(img1,2)
    
    img_l_seq=img1/255.0
    img_h_seq=imgh1/255.0
    for i in range(num_frames-1):
        filei = int(file1) + i + 1
        filenamei = os.path.split(filename)[0] + "/" + "{:>05}".format(filei).format() + ext
        try:
            imgi = imageio.imread(filenamei).astype(np.float32)
            # imgi = sic.imread(filenamei).astype(np.float32)
            imghi = imgi
        except:
            print("Cannot read the following %d frames\n"%(num_frames))
            return None, None
        imgi = cv2.cvtColor(imgi, cv2.COLOR_RGB2GRAY)
        imgi = np.expand_dims(imgi,2)

        img_l_seq = np.concatenate((img_l_seq,imgi/255.0),axis=2)
        img_h_seq = np.concatenate((img_h_seq,imghi/255.0),axis=2)

    return img_l_seq, img_h_seq

# import from favc 
def read_flow_sequence(filename, num_frames):
    file1 = os.path.splitext(os.path.basename(filename))[0]
    folder = os.path.split(filename)[0]
    ext = os.path.splitext(os.path.basename(filename))[1]
    
    filej = file1
    for i in range(num_frames-1):
        filei = int(file1) + i + 1
        if "SPMC" in filename:
            flow_forward = flowlib.read_flow(folder+"/Forward/{:>04}".format(filej).format()+"_"+"{:>04}".format(filei).format()+".flo")
            flow_backward = flowlib.read_flow(folder+"/Backward/{:>04}".format(filei).format()+"_"+"{:>04}".format(filej).format()+".flo")
        else:
            # flow_forward = flowlib.read_flow(folder.replace("480p","Forward")+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".flo")
            # flow_backward = flowlib.read_flow(folder.replace("480p","Backward")+"/"+"{:>05}".format(filei).format()+"_"+"{:>05}".format(filej).format()+".flo")
            flow_forward = flowlib.read_flow(folder+"/Forward"+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".flo")
            # flow_backward = flowlib.read_flow(folder+"/Backward"+"/"+"{:>05}".format(filei).format()+"_"+"{:>05}".format(filej).format()+".flo")
            flow_backward = flowlib.read_flow(folder+"/Backward"+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".flo")

            # print(folder+"/Forward"+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".flo")
            # print(folder+"/Backward"+"/"+"{:>05}".format(filei).format()+"_"+"{:>05}".format(filej).format()+".flo")
            # assert 1==0
        filej = filei
        if i == 0:
            flow_forward_seq = flow_forward
            flow_backward_seq = flow_backward
        else:
            flow_forward_seq = np.concatenate((flow_forward_seq, flow_forward), axis=2)
            flow_backward_seq = np.concatenate((flow_backward_seq, flow_backward), axis=2)

    return flow_forward_seq, flow_backward_seq

# occlusion_mask
def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D

def occlusion_mask(im0, im1, flow10):
    im1 = transforms.ToTensor()(im1).unsqueeze(0)
    # warp_im0 = flow_warp_op.flow_warp(im0, flow10)
    warp_im0 = warp(im0, flow10)
    # print(type(warp_im0))
    warp_im0=warp_im0.cpu()
    # diff = tf.abs(im1 - warp_im0)
    diff = torch.abs(im1 - warp_im0)
    # mask = tf.reduce_max(diff, axis=3, keep_dims=True) #计算一个张量的各个维度上元素的最大值。 
    # mask = tf.less(mask, 0.05)
    # diff_augmax = torch.max(diff, 3, keepdim=True)
    mask = torch.le(diff, 0.05, out=None).int() #以元素方式返回(x <y)的真值.
    # mask = tf.less(diff, 0.05) #以元素方式返回(x <y)的真值.
    # mask = tf.cast(mask, tf.float32) #tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
    mask=torch.FloatTensor(mask.float())
    # mask = tf.transpose(mask, perm=[0,2,3,1])
    mask = mask.permute([0,2,3,1])
    # print(mask.shape)
    mask = mask.repeat(1,1,1,2)
    # mask = tf.tile(mask, [1,1,1,3]) #做成3通道的  每一维数据的扩展都是将前面的数据进行复制然后直接接在原数据后面。
    # print("end occlusion mask")
    return mask, warp_im0

def warp(image2, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    # print(image2.size())
    # 
    # # B, C, H, W = y.size()
    # 
    flo = transforms.ToTensor()(flo).unsqueeze(0)
    # print(flo.shape)  # (436, 1024, 2)
    image2 = transforms.ToTensor()(image2).unsqueeze(0)
    B, C, H, W = image2.shape
    # print(image2.shape, flo.shape)
    # mesh grid 
    xx = torch.arange(0,W).view(1,-1).repeat(H,1)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    
    image2 = image2.cuda()
    flo = flo.cuda()
    grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flo # B,2,H,W
    #图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标

    # scale grid to [-1,1] 
    ##2019 code
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    #取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 #取出光流u这个维度，同上
    # print(type(vgrid))
    vgrid = vgrid.permute(0,2,3,1)#from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
    # output = torch.nn.functional.grid_sample(image2, vgrid, align_corners=True)  #由于pytorch版本较低没有align_corners参数 只能暂时删掉
    output = torch.nn.functional.grid_sample(image2, vgrid)
    mask = torch.autograd.Variable(torch.ones(image2.size())).cuda()
    # mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)#由于pytorch版本较低没有align_corners参数 只能暂时删掉
    mask = torch.nn.functional.grid_sample(mask, vgrid)

    ##2019 author
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    ##2019 code
    # mask = torch.floor(torch.clamp(mask, 0 ,1))

    # # copy tensors to cpu to avoid oom
    # image2 = image2.cpu()
    # flo = flo.cpu()
    # grid = grid.cpu()
    # output = output.cpu()
    # mask = mask.cpu()

    return output*mask


def parse_images(data_root):
    image_pairs = []

    clips = os.listdir(data_root)
    for c_idx, clip in enumerate(clips):
        # img 
        img_names = sorted(os.listdir(os.path.join(data_root, clip, 'pics')))

        for i_idx in range(len(img_names)-1):
            img1_name = os.path.join(data_root, clip, 'pics',img_names[i_idx])
            img2_name = os.path.join(data_root, clip, 'pics',img_names[i_idx+1])
            I_reference_output = os.path.join(data_root, clip, 'pics',img_names[0])
            # I_reference_output = os.path.join(data_root, clip, 'ref', os.listdir(os.path.join(data_root, clip, 'ref'))[0])
            # mask_name = os.path.join(data_root, clip, 'mask', os.listdir(os.path.join(data_root, clip, 'ref'))[0])
            flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
            # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
            flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )
                        
            # print(c_idx, i_idx, clip, img1_name, img2_name, I_reference_output, flow_forward_name, flow_backward_name)

            item = (
                img1_name,
                img2_name,
                I_reference_output,
                flow_forward_name,
                flow_backward_name,
                # mask_name,
            )
            image_pairs.append(item)

    return image_pairs

def parse_images_wBasicVSR(data_root, max_seq_len=None):

    clips = os.listdir(data_root)

    if max_seq_len is None:
        result = []
        for c_idx, clip in enumerate(clips):
            # img 
            img_names = sorted(os.listdir(os.path.join(data_root, clip, 'pics')))

            image_pairs = []
            for i_idx in range(len(img_names)):
                img1_name = os.path.join(data_root, clip, 'pics',img_names[i_idx])
                I_reference_output = os.path.join(data_root, clip, 'pics',img_names[0])
                item = img1_name
                image_pairs.append(item)
            
            result.append(image_pairs)

    else:
        '''
        list:
            result[0] = FANGHUA_V1-001
            result[1] = FANGHUA_V1-002
                result[0][0] = '/home/ysls/Desktop/data/yangyixin/Fanghua_traindata_frame25/FANGHUA_V1-002/pics/0001.png'
                result[0][1] = '/home/ysls/Desktop/data/yangyixin/Fanghua_traindata_frame25/FANGHUA_V1-002/pics/0002.png'

        '''
        result = []
        n_clips = []
        for c_idx, clip in enumerate(clips):
            # img 
            img_names = sorted(os.listdir(os.path.join(data_root, clip, 'pics')))

            image_pairs = []
            for i in range(0, len(img_names), max_seq_len):
                if i == max_seq_len * (len(img_names) // max_seq_len):
                    # corner case: make sure every frame in same length
                    n_image_frame_name = img_names[-max_seq_len:]
                else:
                    n_image_frame_name = img_names[i:i + max_seq_len]
                img1_name = [os.path.join(data_root, clip, 'pics',img_name) for img_name in n_image_frame_name]
                item = img1_name
                image_pairs.append(item)

                # print(i, item)
            # print(len(image_pairs))
            # assert 1==0

            assert len(image_pairs) == math.ceil(len(img_names) / max_seq_len)

            result.append(image_pairs)
            n_clips.append(math.ceil(len(img_names) / max_seq_len))

        # print(len(result))
        # print(len(result[1]))
        # print(n_clips)
        assert len(result)==len(n_clips), 'n_clip中记录的是每个Video切分出来的等长clips的数目'
    return result, n_clips

def parse_images_0618_tcvc_v6(data_root, max_num_sequence, flag_use_precompute_flo=False):
    image_pairs = []
    clips = os.listdir(data_root)
    for c_idx, clip in enumerate(clips):
        # img 
        img_list = []
        img_seg_list = []
        flow_forward_list = []
        flow_backward_list = []
        img_names = sorted(os.listdir(os.path.join(data_root, clip, 'pics')))[:max_num_sequence] if max_num_sequence >= 2 else sorted(os.listdir(os.path.join(data_root, clip, 'pics')))

        Ireference1_name = os.path.join(data_root, clip, 'pics', img_names[0])
        Ireference2_name = os.path.join(data_root, clip, 'pics', img_names[-1])

        for i_idx in range(len(img_names)):
            img1_name = os.path.join(data_root, clip, 'pics', img_names[i_idx])
            img1_seg_name = os.path.join(data_root, clip, 'seg_prop', img_names[i_idx]).split('.')[0]+'.pkl'

            img_list.append(img1_name)
            img_seg_list.append(img1_seg_name)

            if i_idx < len(img_names) - 1 and flag_use_precompute_flo:
                flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
                # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
                flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )

                flow_forward_list.append(flow_forward_name)
                flow_backward_list.append(flow_backward_name)
                        
            # print(c_idx, i_idx, clip, img1_name, img2_name, I_reference_output, flow_forward_name, flow_backward_name)

        item = (
            img_list,
            img_seg_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
            # mask_name,
        )
        image_pairs.append(item)

    return image_pairs, len(img_names)

def parse_images_0618_tcvc(data_root, max_num_sequence, flag_use_precompute_flo=False):
    image_pairs = []
    clips = os.listdir(data_root)
    for c_idx, clip in enumerate(clips):
        # img 
        img_list = []
        flow_forward_list = []
        flow_backward_list = []
        img_names = sorted(os.listdir(os.path.join(data_root, clip, 'pics')))[:max_num_sequence] if max_num_sequence>=2 else sorted(os.listdir(os.path.join(data_root, clip, 'pics')))

        Ireference1_name = os.path.join(data_root, clip, 'pics',img_names[0])
        Ireference2_name = os.path.join(data_root, clip, 'pics',img_names[-1])

        for i_idx in range(len(img_names)):
            img1_name = os.path.join(data_root, clip, 'pics',img_names[i_idx])
            img_list.append(img1_name)

            # print(i_idx, img1_name, img2_name, I_reference_output)
            # assert 1==0 

            if i_idx < len(img_names) - 1 and flag_use_precompute_flo:
                flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
                # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
                flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )

                flow_forward_list.append(flow_forward_name)
                flow_backward_list.append(flow_backward_name)
                        
            # print(c_idx, i_idx, clip, img1_name, img2_name, I_reference_output, flow_forward_name, flow_backward_name)

        item = (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
            # mask_name,
        )
        image_pairs.append(item)

    return image_pairs, len(img_names)

def parse_images_20230227_ntire23(data_root, max_num_sequence, flag_use_precompute_flo=False):
    image_pairs = []
    clips = os.listdir(data_root)
    for c_idx, clip in enumerate(clips):
        # img 
        img_list = []
        flow_forward_list = []
        flow_backward_list = []
        img_names = sorted(os.listdir(os.path.join(data_root, clip)))[:max_num_sequence] if max_num_sequence>=2 else sorted(os.listdir(os.path.join(data_root, clip)))

        Ireference1_name = os.path.join(data_root, clip, img_names[0])
        Ireference2_name = os.path.join(data_root, clip, img_names[-1])

        for i_idx in range(len(img_names)):
            img1_name = os.path.join(data_root, clip, img_names[i_idx])
            img_list.append(img1_name)

            # print(i_idx, img1_name, img2_name, I_reference_output)
            # assert 1==0 

            if i_idx < len(img_names) - 1 and flag_use_precompute_flo:
                flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
                # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
                flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )

                flow_forward_list.append(flow_forward_name)
                flow_backward_list.append(flow_backward_name)
                        
            # print(c_idx, i_idx, clip, img1_name, img2_name, I_reference_output, flow_forward_name, flow_backward_name)

        item = (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
            # mask_name,
        )
        image_pairs.append(item)

    return image_pairs, len(img_names)

class VideosDataset_wBasicVSR(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        max_seq_len = None,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.max_seq_len = max_seq_len
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_clips = parse_images_wBasicVSR(self.data_root, self.max_seq_len)
        self.real_len = len(self.image_pairs)
        print("##### parsing image pairs in %s: %d pairs #####" % (data_root, self.real_len))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        # try:
        
        tmp = self.n_clips.copy()
        tmp.insert(0, 0)
        bin = np.cumsum(tmp)
        
        n_video = np.digitize(index, bin) - 1
        n_video_clip = index - bin[n_video]

        video_clip_names = self.image_pairs[n_video][n_video_clip]


        # print(video_clip_names)
        # assert 1==0

        I_reference_name = self.image_pairs[n_video][0][0]
        I_reference_video = Image.open(I_reference_name)
        # transform
        I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
        
        # print(I_reference_name)
        
        I_all_lab = []
        
        for idx, image1_name in enumerate(video_clip_names):
            # print(image1_name)
            I1 = Image.open(image1_name)
            # transform
            I1 = self.image_transform(self.CenterPad(I1))
            I_all_lab.append(I1.unsqueeze_(dim=0))

        # assert 1==0
        I_all_lab = torch.cat(I_all_lab, dim=0)   

        # print(I_all_lab.shape)
        # assert 1==0 

        I_reference_output = I_reference_video
        placeholder = torch.zeros_like(I_all_lab)
        self_ref_flag = torch.ones_like(I_all_lab)
        # self_ref_flag = self_ref_flag[1:,:,:,:]

        outputs = [
            I_all_lab,
            I_reference_output,
            placeholder,
            self_ref_flag,
        ]

        # except Exception as e:
        #     print("problem in, ", path)
        #     print(e)
        #     return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return sum(self.n_clips)



class VideosDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs = parse_images(self.data_root)
        self.real_len = len(self.image_pairs)
        print("##### parsing image pairs in %s: %d pairs #####" % (data_root, self.real_len))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            image1_name,
            image2_name,
            reference_video_name,
            flow_forward_name,
            flow_backward_name,
            # mask_name,

        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     reference_name1,
        #     reference_name2,
        #     reference_name3,
        #     reference_name4,
        #     reference_name5,
        #     flow_forward_name,
        #     flow_backward_name,
        #     mask_name,
        #     reference_gt1,
        #     reference_gt2,
        #     reference_gt3,
        #     path,
        ) = self.image_pairs[index]
        try:
            I1 = Image.open(image1_name)
            I2 = Image.open(image2_name)

            # print(image1_name, np.shape(I2))

            I_reference_video = Image.open(reference_video_name)
            # I_reference_video = Image.open(image2_name)

            flow_forward = read_flow(flow_forward_name)  # numpy
            flow_backward = read_flow(flow_backward_name)  # numpy

            # calculate occlusion_mask
            cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'),flow_backward)
            cmap_X = cmap_X.squeeze(0)
            cmap_X = cmap_X.numpy()
            mask = cmap_X
            # mask = Image.open(mask_name)

            # binary mask
            mask = np.array(mask)
            # print(np.shape(mask))
            mask[mask < 240] = 0
            mask[mask >= 240] = 1

            # transform
            I1 = self.image_transform(self.CenterPad(I1))
            I2 = self.image_transform(self.CenterPad(I2))
            I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            mask = self.ToTensor(self.CenterCrop(mask))

            I_reference_output = I_reference_video
            placeholder = torch.zeros_like(I1)
            self_ref_flag = torch.ones_like(I1)

            outputs = [
                I1,
                I2,
                I_reference_output,
                flow_forward,
                flow_backward,
                mask,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)

# 0720修改，tcvc_v2版本的基础上，加了segmask
class VideosDataset_0618_tcvc_v6(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        max_num_sequence,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.max_num_sequence = max_num_sequence
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_imgs_pclip = parse_images_0618_tcvc_v6(self.data_root, self.max_num_sequence, self.flag_use_precompute_flo)
        self.real_len = len(self.image_pairs)
        print("##### parsing image clips in %s: %d clips, n_imgs_pclip %s total_iters_perEpoch %s ##### :" % (data_root, len(self.image_pairs), self.n_imgs_pclip, len(self.image_pairs)*self.n_imgs_pclip))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            img_list,
            img_seg_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     flow_forward_name,
        #     flow_backward_name,
        #     # mask_name,
        ) = self.image_pairs[index]
        try:
            # Input Images
            I_list = [Image.open(image_name).convert('RGB') for image_name in img_list]
            I_seg_list = [open(image_seg_name, 'rb') for image_seg_name in img_seg_list]

            # Reference Image
            I1reference_video = Image.open(Ireference1_name).convert('RGB')
            I2reference_video = Image.open(Ireference2_name).convert('RGB')

            # Optical flow 
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

            trans_backward = superslomo_transforms.Compose([superslomo_transforms.Resize([216, 384])])
            # calculate occlusion_mask
            if self.flag_use_precompute_flo:
                mask_list = []
                for i_idx in range(len(I_list)-1):
                    I1 = I_list[i_idx]
                    I2 = I_list[i_idx+1]

                    cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'), flow_backward[i_idx])
                    cmap_X = cmap_X.squeeze(0)
                    cmap_X = cmap_X.numpy()
                    mask = cmap_X
                    # mask = Image.open(mask_name)

                    # binary mask
                    mask = np.array(mask)
                    # print(np.shape(mask))
                    mask[mask < 240] = 0
                    mask[mask >= 240] = 1

                    mask_list.append(mask)

            # transform
            I_list = [self.image_transform(self.CenterPad(I1)) for I1 in I_list]
            I_seg_list = [trans_backward(torch.from_numpy(np.array(pickle.load(I1_seg)))[0:21, :, :].type(torch.FloatTensor)) for I1_seg in I_seg_list]

            I1reference_video = self.image_transform(self.CenterPad(I1reference_video))
            I2reference_video = self.image_transform(self.CenterPad(I2reference_video))
            flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            placeholder = torch.zeros_like(I_list[0])
            self_ref_flag = torch.ones_like(I_list[0])

            # I1 = self.image_transform(self.CenterPad(I1))
            # I2 = self.image_transform(self.CenterPad(I2))
            # I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            # flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            # flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            # mask = self.ToTensor(self.CenterCrop(mask))

            # I_reference_output = I_reference_video
            # placeholder = torch.zeros_like(I1)
            # self_ref_flag = torch.ones_like(I1)

            outputs = [
                I_list,
                I_seg_list,
                I1reference_video,
                I2reference_video,
                flow_forward,
                flow_backward,
                mask_list,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len


# 0720修改，tcvc_v2版本的基础上，加了segmask
class VideosDataset_0618_tcvc_v7(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        max_num_sequence,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.max_num_sequence = max_num_sequence
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        self.image_size = image_size

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_imgs_pclip = parse_images_0618_tcvc_v6(self.data_root, self.max_num_sequence, self.flag_use_precompute_flo)
        self.real_len = len(self.image_pairs)
        print("##### parsing image clips in %s: %d clips, n_imgs_pclip %s total_iters_perEpoch %s ##### :" % (data_root, len(self.image_pairs), self.n_imgs_pclip, len(self.image_pairs)*self.n_imgs_pclip))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            img_list,
            img_seg_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
        ) = self.image_pairs[index]
        try:
            ### define transforms based on RandomCrop
            self_RandomCrop = RandomCrop(self.image_size)
            transforms_video_ref = [
                self_RandomCrop,
                RGB2Lab(),
                ToTensor(),
                Normalize(),
            ]
            image_transform_ref = train_transforms.Compose(transforms_video_ref)


            # Input Images
            I_list = [Image.open(image_name).convert('RGB') for image_name in img_list]
            I_seg_list = [open(image_seg_name, 'rb') for image_seg_name in img_seg_list]

            # Reference Image
            I1reference_video = Image.open(Ireference1_name).convert('RGB')
            I2reference_video = Image.open(Ireference2_name).convert('RGB')

            # Optical flow 
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

            # calculate occlusion_mask
            if self.flag_use_precompute_flo:
                mask_list = []
                for i_idx in range(len(I_list)-1):
                    I1 = I_list[i_idx]
                    I2 = I_list[i_idx+1]

                    cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'), flow_backward[i_idx])
                    cmap_X = cmap_X.squeeze(0)
                    cmap_X = cmap_X.numpy()
                    mask = cmap_X
                    # mask = Image.open(mask_name)

                    # binary mask
                    mask = np.array(mask)
                    # print(np.shape(mask))
                    mask[mask < 240] = 0
                    mask[mask >= 240] = 1

                    mask_list.append(mask)

            # transform
            I_list = [image_transform_ref(I1) for I1 in I_list]
            I_seg_list = [self_RandomCrop(torch.from_numpy(np.array(pickle.load(I1_seg)))[0:21, :, :].type(torch.FloatTensor)) for I1_seg in I_seg_list]
            I1reference_video = image_transform_ref(I1reference_video)
            I2reference_video = image_transform_ref(I2reference_video)

            # print(I_list[0].shape, I_seg_list[0].shape, np.shape(flow_forward[0]));assert 1==0

            flow_forward = [self_RandomCrop(self.ToTensor(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            flow_backward = [self_RandomCrop(self.ToTensor(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            mask_list = [self_RandomCrop(self.ToTensor(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            # flow_forward = [self.ToTensor(self_RandomCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            # flow_backward = [self.ToTensor(self_RandomCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            # mask_list = [self.ToTensor(self_RandomCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            
            placeholder = torch.zeros_like(I_list[0])
            self_ref_flag = torch.ones_like(I_list[0])


            # I_list = [self.image_transform(self.CenterPad(I1)) for I1 in I_list]
            # I_seg_list = [trans_backward(torch.from_numpy(np.array(pickle.load(I1_seg)))[0:21, :, :].type(torch.FloatTensor)) for I1_seg in I_seg_list]
            # I1reference_video = self.image_transform(self.CenterPad(I1reference_video))
            # I2reference_video = self.image_transform(self.CenterPad(I2reference_video))
            # flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            # flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            # mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            # placeholder = torch.zeros_like(I_list[0])
            # self_ref_flag = torch.ones_like(I_list[0])


            outputs = [
                I_list,
                I_seg_list,
                I1reference_video,
                I2reference_video,
                flow_forward,
                flow_backward,
                mask_list,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len

# 1005修改版本，dataloader处理修改，使用重复epoch v2
def parse_images_0930_DavidVideovo_repeatEpoch(data_root, max_num_sequence, flag_use_precompute_flo=False, epoch=4000):

    image_pairs = []
    clips = os.listdir(data_root)

    for c_idx, clip in enumerate(clips):
        # generate random number
        RandomNum = random.randint(0,len(os.listdir(os.path.join(data_root, clip)))-max_num_sequence)

        # img 
        img_list = []
        flow_forward_list = []
        flow_backward_list = []
        # img_names = sorted(os.listdir(os.path.join(data_root, clip)))[:max_num_sequence] if max_num_sequence>=2 else sorted(os.listdir(os.path.join(data_root, clip, 'pics')))
        img_names = sorted(os.listdir(os.path.join(data_root, clip)))[RandomNum : RandomNum + max_num_sequence] if max_num_sequence>=2 else sorted(os.listdir(os.path.join(data_root, clip, 'pics')))
        
        # debug code 
        # print(idx_epoch, c_idx, clip, RandomNum, img_names)

        Ireference1_name = os.path.join(data_root, clip, img_names[0])
        Ireference2_name = os.path.join(data_root, clip, img_names[-1])

        for i_idx in range(len(img_names)):
            img1_name = os.path.join(data_root, clip, img_names[i_idx])
            img_list.append(img1_name)

            # print(i_idx, img1_name, img2_name, I_reference_output)
            # assert 1==0 

            if i_idx < len(img_names) - 1 and flag_use_precompute_flo:
                flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
                # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
                flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )

                flow_forward_list.append(flow_forward_name)
                flow_backward_list.append(flow_backward_name)
                        
            # print(c_idx, i_idx, clip, img1_name, img2_name, I_reference_output, flow_forward_name, flow_backward_name)

        item = (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
            # mask_name,
        )
        image_pairs.append(item)

    return image_pairs, len(img_names)



def parse_images_0930_DavidVideovo(data_root, max_num_sequence, flag_use_precompute_flo=False, epoch=4000):

    image_pairs = []
    clips = os.listdir(data_root)
    for idx_epoch in range(epoch):
        for c_idx, clip in enumerate(clips):
            # generate random number
            RandomNum = random.randint(0,len(os.listdir(os.path.join(data_root, clip)))-max_num_sequence)

            # img 
            img_list = []
            flow_forward_list = []
            flow_backward_list = []
            # img_names = sorted(os.listdir(os.path.join(data_root, clip)))[:max_num_sequence] if max_num_sequence>=2 else sorted(os.listdir(os.path.join(data_root, clip, 'pics')))
            img_names = sorted(os.listdir(os.path.join(data_root, clip)))[RandomNum : RandomNum + max_num_sequence] if max_num_sequence>=2 else sorted(os.listdir(os.path.join(data_root, clip, 'pics')))
            
            # debug code 
            # print(idx_epoch, c_idx, clip, RandomNum, img_names)

            Ireference1_name = os.path.join(data_root, clip, img_names[0])
            Ireference2_name = os.path.join(data_root, clip, img_names[-1])

            for i_idx in range(len(img_names)):
                img1_name = os.path.join(data_root, clip, img_names[i_idx])
                img_list.append(img1_name)

                # print(i_idx, img1_name, img2_name, I_reference_output)
                # assert 1==0 

                if i_idx < len(img_names) - 1 and flag_use_precompute_flo:
                    flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
                    # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
                    flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )

                    flow_forward_list.append(flow_forward_name)
                    flow_backward_list.append(flow_backward_name)
                            
                # print(c_idx, i_idx, clip, img1_name, img2_name, I_reference_output, flow_forward_name, flow_backward_name)

            item = (
                img_list,
                Ireference1_name,
                Ireference2_name,
                flow_forward_list,
                flow_backward_list,
                # mask_name,
            )
            image_pairs.append(item)

    return image_pairs, len(img_names)



# 1005修改，原型VideosDataset_0618_tcvc， 2参考帧， DAVIS+VIDEOVO 修改了dataloader，每个epoch内部重复
class VideosDataset_0930_DavisVideovo_repeatEpoch(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        max_num_sequence,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.max_num_sequence = max_num_sequence
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_imgs_pclip = parse_images_0930_DavidVideovo_repeatEpoch(self.data_root, self.max_num_sequence, self.flag_use_precompute_flo, self.epoch)
        self.real_len = len(self.image_pairs)
        print("##### parsing image clips in %s: %d clips, n_imgs_pclip %s total_iters_perEpoch %s ##### :" % (data_root, len(self.image_pairs), self.n_imgs_pclip, len(self.image_pairs)))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     flow_forward_name,
        #     flow_backward_name,
        #     # mask_name,
        ) = self.image_pairs[index]
        try:
            # Input Images
            I_list = [Image.open(image_name).convert('RGB') for image_name in img_list]

            # Reference Image
            I1reference_video = Image.open(Ireference1_name).convert('RGB')
            I2reference_video = Image.open(Ireference2_name).convert('RGB')


            # Optical flow 
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

            # calculate occlusion_mask
            if self.flag_use_precompute_flo:
                mask_list = []
                for i_idx in range(len(I_list)-1):
                    I1 = I_list[i_idx]
                    I2 = I_list[i_idx+1]

                    # print(len(I_list), type(I1));assert 1==0

                    cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'), flow_backward[i_idx])
                    cmap_X = cmap_X.squeeze(0)
                    cmap_X = cmap_X.numpy()
                    mask = cmap_X
                    # mask = Image.open(mask_name)

                    # binary mask
                    mask = np.array(mask)
                    # print(np.shape(mask))
                    mask[mask < 240] = 0
                    mask[mask >= 240] = 1

                    mask_list.append(mask)

            # transform
            I_list = [self.image_transform(self.CenterPad(I1)) for I1 in I_list]
            I1reference_video = self.image_transform(self.CenterPad(I1reference_video))
            I2reference_video = self.image_transform(self.CenterPad(I2reference_video))
            flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            placeholder = torch.zeros_like(I_list[0])
            self_ref_flag = torch.ones_like(I_list[0])

            # I1 = self.image_transform(self.CenterPad(I1))
            # I2 = self.image_transform(self.CenterPad(I2))
            # I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            # flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            # flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            # mask = self.ToTensor(self.CenterCrop(mask))

            # I_reference_output = I_reference_video
            # placeholder = torch.zeros_like(I1)
            # self_ref_flag = torch.ones_like(I1)

            outputs = [
                I_list,
                I1reference_video,
                I2reference_video,
                flow_forward,
                flow_backward,
                mask_list,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len


# 0930修改，原型VideosDataset_0618_tcvc， 2参考帧， DAVIS+VIDEOVO
class VideosDataset_0930_DavisVideovo(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        max_num_sequence,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.max_num_sequence = max_num_sequence
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_imgs_pclip = parse_images_0930_DavidVideovo(self.data_root, self.max_num_sequence, self.flag_use_precompute_flo, self.epoch)
        self.real_len = len(self.image_pairs)
        print("##### parsing image clips in %s: %d clips, n_imgs_pclip %s total_iters_perEpoch %s ##### :" % (data_root, len(self.image_pairs), self.n_imgs_pclip, len(self.image_pairs)))
        # self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     flow_forward_name,
        #     flow_backward_name,
        #     # mask_name,
        ) = self.image_pairs[index]
        try:
            # Input Images
            I_list = [Image.open(image_name).convert('RGB') for image_name in img_list]

            # Reference Image
            I1reference_video = Image.open(Ireference1_name).convert('RGB')
            I2reference_video = Image.open(Ireference2_name).convert('RGB')


            # Optical flow 
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

            # calculate occlusion_mask
            if self.flag_use_precompute_flo:
                mask_list = []
                for i_idx in range(len(I_list)-1):
                    I1 = I_list[i_idx]
                    I2 = I_list[i_idx+1]

                    # print(len(I_list), type(I1));assert 1==0

                    cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'), flow_backward[i_idx])
                    cmap_X = cmap_X.squeeze(0)
                    cmap_X = cmap_X.numpy()
                    mask = cmap_X
                    # mask = Image.open(mask_name)

                    # binary mask
                    mask = np.array(mask)
                    # print(np.shape(mask))
                    mask[mask < 240] = 0
                    mask[mask >= 240] = 1

                    mask_list.append(mask)

            # transform
            I_list = [self.image_transform(self.CenterPad(I1)) for I1 in I_list]
            I1reference_video = self.image_transform(self.CenterPad(I1reference_video))
            I2reference_video = self.image_transform(self.CenterPad(I2reference_video))
            flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            placeholder = torch.zeros_like(I_list[0])
            self_ref_flag = torch.ones_like(I_list[0])

            # I1 = self.image_transform(self.CenterPad(I1))
            # I2 = self.image_transform(self.CenterPad(I2))
            # I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            # flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            # flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            # mask = self.ToTensor(self.CenterCrop(mask))

            # I_reference_output = I_reference_video
            # placeholder = torch.zeros_like(I1)
            # self_ref_flag = torch.ones_like(I1)

            outputs = [
                I_list,
                I1reference_video,
                I2reference_video,
                flow_forward,
                flow_backward,
                mask_list,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len


# 20230227修改，ntire23
class VideosDataset_20230227_ntire23(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        max_num_sequence,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.max_num_sequence = max_num_sequence
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_imgs_pclip = parse_images_20230227_ntire23(self.data_root, self.max_num_sequence, self.flag_use_precompute_flo)
        self.real_len = len(self.image_pairs)
        print("##### parsing image clips in %s: %d clips, n_imgs_pclip %s total_iters_perEpoch %s ##### :" % (data_root, len(self.image_pairs), self.n_imgs_pclip, len(self.image_pairs)*self.n_imgs_pclip))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     flow_forward_name,
        #     flow_backward_name,
        #     # mask_name,
        ) = self.image_pairs[index]
        try:
            # Input Images
            I_list = [Image.open(image_name).convert('RGB') for image_name in img_list]

            # Reference Image
            I1reference_video = Image.open(Ireference1_name).convert('RGB')
            I2reference_video = Image.open(Ireference2_name).convert('RGB')


            # Optical flow 
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

            # calculate occlusion_mask
            if self.flag_use_precompute_flo:
                mask_list = []
                for i_idx in range(len(I_list)-1):
                    I1 = I_list[i_idx]
                    I2 = I_list[i_idx+1]

                    # print(len(I_list), type(I1));assert 1==0

                    cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'), flow_backward[i_idx])
                    cmap_X = cmap_X.squeeze(0)
                    cmap_X = cmap_X.numpy()
                    mask = cmap_X
                    # mask = Image.open(mask_name)

                    # binary mask
                    mask = np.array(mask)
                    # print(np.shape(mask))
                    mask[mask < 240] = 0
                    mask[mask >= 240] = 1

                    mask_list.append(mask)

            # transform
            I_list = [self.image_transform(self.CenterPad(I1)) for I1 in I_list]
            I1reference_video = self.image_transform(self.CenterPad(I1reference_video))
            I2reference_video = self.image_transform(self.CenterPad(I2reference_video))
            flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            placeholder = torch.zeros_like(I_list[0])
            self_ref_flag = torch.ones_like(I_list[0])

            # I1 = self.image_transform(self.CenterPad(I1))
            # I2 = self.image_transform(self.CenterPad(I2))
            # I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            # flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            # flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            # mask = self.ToTensor(self.CenterCrop(mask))

            # I_reference_output = I_reference_video
            # placeholder = torch.zeros_like(I1)
            # self_ref_flag = torch.ones_like(I1)

            outputs = [
                I_list,
                I1reference_video,
                I2reference_video,
                flow_forward,
                flow_backward,
                mask_list,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len



# 0618修改，视频序列首尾两帧作为参考帧进行训练
class VideosDataset_0618_tcvc(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        max_num_sequence,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.max_num_sequence = max_num_sequence
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs, self.n_imgs_pclip = parse_images_0618_tcvc(self.data_root, self.max_num_sequence, self.flag_use_precompute_flo)
        self.real_len = len(self.image_pairs)
        print("##### parsing image clips in %s: %d clips, n_imgs_pclip %s total_iters_perEpoch %s ##### :" % (data_root, len(self.image_pairs), self.n_imgs_pclip, len(self.image_pairs)*self.n_imgs_pclip))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            img_list,
            Ireference1_name,
            Ireference2_name,
            flow_forward_list,
            flow_backward_list,
        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     flow_forward_name,
        #     flow_backward_name,
        #     # mask_name,
        ) = self.image_pairs[index]
        try:
            # Input Images
            I_list = [Image.open(image_name).convert('RGB') for image_name in img_list]

            # Reference Image
            I1reference_video = Image.open(Ireference1_name).convert('RGB')
            I2reference_video = Image.open(Ireference2_name).convert('RGB')


            # Optical flow 
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

            # calculate occlusion_mask
            if self.flag_use_precompute_flo:
                mask_list = []
                for i_idx in range(len(I_list)-1):
                    I1 = I_list[i_idx]
                    I2 = I_list[i_idx+1]

                    # print(len(I_list), type(I1));assert 1==0

                    cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'), flow_backward[i_idx])
                    cmap_X = cmap_X.squeeze(0)
                    cmap_X = cmap_X.numpy()
                    mask = cmap_X
                    # mask = Image.open(mask_name)

                    # binary mask
                    mask = np.array(mask)
                    # print(np.shape(mask))
                    mask[mask < 240] = 0
                    mask[mask >= 240] = 1

                    mask_list.append(mask)

            # transform
            I_list = [self.image_transform(self.CenterPad(I1)) for I1 in I_list]
            I1reference_video = self.image_transform(self.CenterPad(I1reference_video))
            I2reference_video = self.image_transform(self.CenterPad(I2reference_video))
            flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else []
            flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else []
            mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else []
            placeholder = torch.zeros_like(I_list[0])
            self_ref_flag = torch.ones_like(I_list[0])

            # I1 = self.image_transform(self.CenterPad(I1))
            # I2 = self.image_transform(self.CenterPad(I2))
            # I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            # flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            # flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            # mask = self.ToTensor(self.CenterCrop(mask))

            # I_reference_output = I_reference_video
            # placeholder = torch.zeros_like(I1)
            # self_ref_flag = torch.ones_like(I1)

            outputs = [
                I_list,
                I1reference_video,
                I2reference_video,
                flow_forward,
                flow_backward,
                mask_list,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len

class VideosDataset_0626_video_interpolation(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs = parse_images(self.data_root)
        self.real_len = len(self.image_pairs)
        print("##### parsing image pairs in %s: %d pairs #####" % (data_root, self.real_len))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            image1_name,
            image2_name,
            reference_video_name,
            flow_forward_name,
            flow_backward_name,
            # mask_name,

        #     image1_name,
        #     image2_name,
        #     reference_video_name,
        #     reference_name1,
        #     reference_name2,
        #     reference_name3,
        #     reference_name4,
        #     reference_name5,
        #     flow_forward_name,
        #     flow_backward_name,
        #     mask_name,
        #     reference_gt1,
        #     reference_gt2,
        #     reference_gt3,
        #     path,
        ) = self.image_pairs[index]
        try:
            I1 = Image.open(image1_name)
            I2 = Image.open(image2_name)

            # print(image1_name, np.shape(I2))

            I_reference_video = Image.open(reference_video_name)
            # I_reference_video = Image.open(image2_name)

            flow_forward = read_flow(flow_forward_name)  # numpy
            flow_backward = read_flow(flow_backward_name)  # numpy

            # calculate occlusion_mask
            cmap_X, warp_X0 = occlusion_mask(I1.convert('L') ,I2.convert('L'),flow_backward)
            cmap_X = cmap_X.squeeze(0)
            cmap_X = cmap_X.numpy()
            mask = cmap_X
            # mask = Image.open(mask_name)

            # binary mask
            mask = np.array(mask)
            # print(np.shape(mask))
            mask[mask < 240] = 0
            mask[mask >= 240] = 1

            # transform
            I1 = self.image_transform(self.CenterPad(I1))
            I2 = self.image_transform(self.CenterPad(I2))
            I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            mask = self.ToTensor(self.CenterCrop(mask))

            I_reference_output = I_reference_video

            # placeholder = torch.zeros_like(I1)
            placeholder = torch.zeros_like(I1)
            placeholder = torch.cat([placeholder[1:,:,:], placeholder, placeholder], dim = 0)

            self_ref_flag = torch.ones_like(I1)

            outputs = [
                I1,
                I2,
                I_reference_output,
                flow_forward,
                flow_backward,
                mask,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)


def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow=8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    l_norm, ab_norm = 1.0, 1.0
    l_mean, ab_mean = 50.0, 0
    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")
