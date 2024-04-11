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

import lib.functional as F

import torchvision.transforms as transforms

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
        elif input_numpy.ndim == 2:
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


def parse_images(data_root):
    image_pairs = []
    subdirs = sorted(os.listdir(data_root))
    for subdir in subdirs:
        path = os.path.join(data_root, subdir)
        if not os.path.isdir(path):
            continue

        parse_file = os.path.join(path, "pairs_output_new.txt")
        if os.path.exists(parse_file):
            with open(parse_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    (
                        image1_name,
                        image2_name,
                        reference_video_name,
                        reference_video_name1,
                        reference_name1,
                        reference_name2,
                        reference_name3,
                        reference_name4,
                        reference_name5,
                        reference_gt1,
                        reference_gt2,
                        reference_gt3,
                    ) = line.split()
                    image1_name = image1_name.split(".")[0]
                    image2_name = image2_name.split(".")[0]
                    reference_video_name = reference_video_name.split(".")[0]
                    reference_video_name1 = reference_video_name1.split(".")[0]
                    reference_name1 = reference_name1.split(".")[0]
                    reference_name2 = reference_name2.split(".")[0]
                    reference_name3 = reference_name3.split(".")[0]
                    reference_name4 = reference_name4.split(".")[0]
                    reference_name5 = reference_name5.split(".")[0]

                    reference_gt1 = reference_gt1.split(".")[0]
                    reference_gt2 = reference_gt2.split(".")[0]
                    reference_gt3 = reference_gt3.split(".")[0]

                    flow_forward_name = image1_name + "_forward"
                    flow_backward_name = image1_name + "_backward"
                    mask_name = image1_name + "_mask"

                    item = (
                        image1_name + ".jpg",
                        image2_name + ".jpg",
                        reference_video_name + ".jpg",
                        reference_name1 + ".JPEG",
                        reference_name2 + ".JPEG",
                        reference_name3 + ".JPEG",
                        reference_name4 + ".JPEG",
                        reference_name5 + ".JPEG",
                        flow_forward_name + ".flo",
                        flow_backward_name + ".flo",
                        mask_name + ".pgm",
                        reference_gt1 + ".jpg",
                        reference_gt2 + ".jpg",
                        reference_gt3 + ".jpg",
                        path,
                    )
                    image_pairs.append(item)

        else:
            raise (RuntimeError("Error when parsing pair_output_count.txt in subfolders of: " + path + "\n"))

    return image_pairs

def parse_images_20230227_ntire23(data_root, flag_use_precompute_flo=False):
    image_pairs = []
    clips = os.listdir(data_root)
    for c_idx, clip in enumerate(clips):
        # img 
        flow_forward_list = []
        flow_backward_list = []
        img_names = sorted(os.listdir(os.path.join(data_root, clip)))

        Ireference1_name = os.path.join(data_root, clip, img_names[0])

        for i_idx in range(len(img_names)-1):
            img1_name = os.path.join(data_root, clip, img_names[i_idx])
            img2_name = os.path.join(data_root, clip, img_names[i_idx+1])

            if i_idx < len(img_names) - 1 and flag_use_precompute_flo:
                flow_forward_name = os.path.join(data_root, clip, 'flo', 'Forward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' ) 
                # flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx].split(".")[0] + '_' + img_names[i_idx+1].split(".")[0] + '.flo' )
                flow_backward_name = os.path.join(data_root, clip, 'flo', 'Backward', img_names[i_idx+1].split(".")[0] + '_' + img_names[i_idx].split(".")[0] + '.flo' )

                flow_forward_list.append(flow_forward_name)
                flow_backward_list.append(flow_backward_name)

        item = (
            img1_name,
            img2_name,
            Ireference1_name,
            flow_forward_list,
            flow_backward_list,
        )
        image_pairs.append(item)

    return image_pairs

class VideosDataset_20230227_ntire23(torch.utils.data.Dataset):
    def __init__(
        self,
        flag_use_precompute_flo,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        flow_input_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
        raft_model=None
    ):
        self.raft_model = raft_model
        self.flag_use_precompute_flo = flag_use_precompute_flo
        self.data_root = data_root
        self.image_transform = image_transform
        self.flow_input_transform = flow_input_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs = parse_images_20230227_ntire23(self.data_root)
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
            flow_forward_list,
            flow_backward_list,
        ) = self.image_pairs[index]

        # Input Images
        I1 = Image.open(image1_name).convert('RGB')
        I2 = Image.open(image2_name).convert('RGB')
        I_reference_video = Image.open(reference_video_name).convert('RGB')

        # transform Flow input images
        I1_flowinput = self.flow_input_transform(I1).cuda()
        I2_flowinput = self.flow_input_transform(I2).cuda()
        I1_flowinput = I1_flowinput / 255.
        I2_flowinput = I2_flowinput / 255.
        
        if not self.flag_use_precompute_flo:
            with torch.no_grad():
                flow_forward, flow_backward = self.compute_flow(I1_flowinput, I2_flowinput, flag_save_flow_warp=False)
        else:
            # read precomputed flow
            flow_forward = [read_flow(flow_forward_name) for flow_forward_name in flow_forward_list] if self.flag_use_precompute_flo else []
            flow_backward = [read_flow(flow_backward_name) for flow_backward_name in flow_forward_list] if self.flag_use_precompute_flo else []

        # calculate occlusion_mask
        cmap_X, warp_X0 = self.occlusion_mask(I1_flowinput.unsqueeze(0), I2_flowinput.unsqueeze(0), flow_backward)
        cmap_X = cmap_X.squeeze(0)
        cmap_X = cmap_X.cpu().numpy()
        mask = cmap_X
        # binary mask
        mask = np.array(mask)
        mask[mask < 240] = 0
        mask[mask >= 240] = 1

        # transform
        I1_tensor = self.image_transform(self.CenterPad(I1)).cuda()
        I2_tensor = self.image_transform(self.CenterPad(I2)).cuda()
        I_reference_video_tensor = self.image_transform(self.CenterPad(I_reference_video))
        flow_forward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_forward] if self.flag_use_precompute_flo else flow_forward.cuda()
        flow_backward = [self.ToTensor(self.CenterCrop(flow)) for flow in flow_backward] if self.flag_use_precompute_flo else flow_backward.cuda()
        mask_list = [self.ToTensor(self.CenterCrop(mask)) for mask in mask_list] if self.flag_use_precompute_flo else self.ToTensor(self.CenterCrop(mask))
        placeholder = torch.zeros_like(I1_tensor).cpu()
        self_ref_flag = torch.ones_like(I1_tensor).cpu()

        outputs = [
            I1_tensor.cpu(),
            I2_tensor.cpu(),
            I_reference_video_tensor,
            flow_forward.cpu(),
            flow_backward.cpu(),
            mask_list,
            placeholder,
            self_ref_flag,
        ]

        # except Exception as e:
        #     print("problem in, ", Ireference1_name)
        #     print(e)
        #     return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return self.real_len

    @torch.no_grad()
    def compute_flow(self, image1, image2, flag_save_flow_warp):
        raft = self.raft_model
        c, h, w = image1.size()

        with torch.no_grad():
            image1 = image1.unsqueeze(0) * 255.
            image2 = image2.unsqueeze(0) * 255.            
            # image1 = image1.unsqueeze(0)
            # image2 = image2.unsqueeze(0)   

            flow_low, flow_forward = raft(image2, image1, iters=20, test_mode=True)
            flow_low, flow_backward = raft(image1, image2, iters=20, test_mode=True)

        return flow_forward, flow_backward
    
    @torch.no_grad()
    def occlusion_mask(self, im0, im1, flow10):
        # im1 = transforms.ToTensor()(im1).unsqueeze(0)
        warp_im0 = self.warp(im0, flow10)
        # warp_im0=warp_im0.cpu()
        diff = torch.abs(im1 - warp_im0)
        mask = torch.le(diff, 0.05, out=None).int() 
        # mask=torch.FloatTensor(mask.float())
        mask=mask.float()
        mask = mask.permute([0,2,3,1])
        mask = mask.repeat(1,1,1,2)
        return mask, warp_im0
    
    @torch.no_grad()
    def warp(self, image2, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        # flo = transforms.ToTensor()(flo).unsqueeze(0)
        # print(flo.shape)  # (436, 1024, 2)
        # image2 = transforms.ToTensor()(image2).unsqueeze(0)
        B, C, H, W = image2.shape

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

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 #取出光流u这个维度，同上
        vgrid = vgrid.permute(0,2,3,1)
        output = torch.nn.functional.grid_sample(image2, vgrid)
        mask = torch.autograd.Variable(torch.ones(image2.size())).cuda()
        mask = torch.nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask


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
            reference_name1,
            reference_name2,
            reference_name3,
            reference_name4,
            reference_name5,
            flow_forward_name,
            flow_backward_name,
            mask_name,
            reference_gt1,
            reference_gt2,
            reference_gt3,
            path,
        ) = self.image_pairs[index]
        try:
            I1 = Image.open(os.path.join(path, "input_pad", image1_name))
            I2 = Image.open(os.path.join(path, "input_pad", image2_name))

            I_reference_video = Image.open(
                os.path.join(path, "reference_gt", random.choice([reference_gt1, reference_gt2, reference_gt3]))
            )
            I_reference_video_real = Image.open(
                os.path.join(
                    path,
                    "reference",
                    random.choice(
                        [reference_name1, reference_name2, reference_name3, reference_name4, reference_name5]
                    ),
                )
            )

            flow_forward = read_flow(os.path.join(path, "flow", flow_forward_name))  # numpy
            flow_backward = read_flow(os.path.join(path, "flow", flow_backward_name))  # numpy
            mask = Image.open(os.path.join(path, "mask", mask_name))

            # binary mask
            mask = np.array(mask)
            mask[mask < 240] = 0
            mask[mask >= 240] = 1

            # transform
            I1 = self.image_transform(I1)
            I2 = self.image_transform(I2)
            I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            I_reference_video_real = self.image_transform(self.CenterPad(I_reference_video_real))
            flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            mask = self.ToTensor(self.CenterCrop(mask))

            if np.random.random() < self.real_reference_probability:
                I_reference_output = I_reference_video_real
                placeholder = torch.zeros_like(I1)
                self_ref_flag = torch.zeros_like(I1)
            else:
                I_reference_output = I_reference_video
                placeholder = I2 if np.random.random() < self.nonzero_placeholder_probability else torch.zeros_like(I1)
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
