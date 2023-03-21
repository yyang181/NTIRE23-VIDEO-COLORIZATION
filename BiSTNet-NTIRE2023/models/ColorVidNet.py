import torch
import torch.nn as nn
import torch.nn.parallel

import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class ATB_block(nn.Module):
    def __init__(self,channel=2):
        super(ATB_block, self).__init__()
        self.channel = channel
        self.atb_forward = ATB(self.channel)
        self.atb_backward = ATB(self.channel)
        self.atb_fuse = ATB(self.channel)
        self.atb_fuse_videointerp_atb = ATB(self.channel)

    def forward(self, colorvid1, colorvid2, flows_forward, flows_backward):
        n, c, h, w = colorvid1[0].size()
        t = len(colorvid1)
        # print(n,t,c,h,w);assert 1==0

        ab_fuse_videointerp = []
        ab_fuse_atb = []
        for i_idx in range(t):
            # print('**************************************  %s **********************************'%i_idx)
            a = colorvid1[i_idx][:,1:,:,:]
            b = colorvid2[i_idx][:,1:,:,:]

            t0 = 1 / (t-1)
            ti = i_idx * t0
            # I_current_ab_predict = (1-t)*a + t*b
            ab_fuse_videointerp.append((1-ti)*a + ti*b)

            # fused = self.atb_fuse_videointerp_atb(colorvid1[i_idx][:,1:,:,:], colorvid2[i_idx][:,1:,:,:])
            fused = self.atb_fuse_videointerp_atb(a, b)
            # ab_fuse_atb.append(fused.detach())
            ab_fuse_atb.append(fused)

        # backward-time propgation
        backward_propagation = []
        backward_propagation.append(colorvid2[-1][:,1:,:,:])
        lr_curr = colorvid2[-1][:,1:,:,:]
        for i in range(t - 2, -1, -1):
            flow = flows_backward[i]
            feat_prop = flow_warp(lr_curr, flow.permute(0, 2, 3, 1), padding_mode='border')
            feat_atb = self.atb_backward(colorvid2[i][:,1:,:,:], feat_prop)

            lr_curr = feat_atb
            backward_propagation.append(feat_atb)
        backward_propagation.reverse()

        # forward-time propagation and upsampling
        result = []
        result.append(colorvid1[0][:,1:,:,:])
        forward_propagation = []
        forward_propagation.append(colorvid1[0][:,1:,:,:])
        lr_curr = colorvid1[0][:,1:,:,:]
        for i in range(0, t-1):
            flow = flows_forward[i]
            feat_prop = flow_warp(lr_curr, flow.permute(0, 2, 3, 1), padding_mode='border')
            feat_atb = self.atb_forward(colorvid1[i+1][:,1:,:,:], feat_prop)

            lr_curr = feat_atb
            forward_propagation.append(feat_atb)

            if i < t-2:
                # print(i, len(backward_propagation))
                feat_fuse = self.atb_fuse(feat_atb, backward_propagation[i+1])
                # result.append(feat_fuse.detach())
                result.append(feat_fuse)

        result.append(colorvid2[-1][:,1:,:,:])


        # combine ab_fuse_videointerp, ab_fuse_atb and result
        result_fuse = []
        for i_idx in range(t):
            result_fuse_i = torch.cat([result[i_idx], ab_fuse_videointerp[i_idx], ab_fuse_atb[i_idx]], dim=1)
            result_fuse.append(result_fuse_i)


        return result_fuse, ab_fuse_videointerp, ab_fuse_atb



class ATB(nn.Module):
    def __init__(self,channel=64):
        super(ATB, self).__init__()
        self.channel = channel
        self.ATB_preconv = nn.Conv2d(self.channel*2, self.channel*2, 3, 1, 1)
        self.ATB_11 = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        self.ATB_12 = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        self.ATB_21 = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        self.ATB_22 = nn.Conv2d(self.channel, self.channel, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea_x, fea_ref):
        feat_concat = torch.cat([fea_x, fea_ref], dim=1)
        feat_fus = self.ATB_preconv(feat_concat)
        feat_split = torch.split(feat_fus, [self.channel, self.channel], dim=1)
        # up branch
        feat_up = self.sigmoid(self.ATB_12(self.lrelu(self.ATB_11(feat_split[0]))))
        feat_up = fea_ref * feat_up
        # down branch
        feat_down = self.sigmoid(self.ATB_21(self.lrelu(self.ATB_22(feat_split[1]))))
        feat_down = fea_x * feat_down
        # sum
        feat_prop = feat_up + feat_down
        return feat_prop


# 0711 v2: with ATB 
class ColorVidNet_wBasicVSR_v2(nn.Module):
    def __init__(self, ic, flag_propagation = False, mid_channels = 64):
        super(ColorVidNet_wBasicVSR_v2, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.BatchNorm2d(64, affine=False)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv9_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv10_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.flag_propagation = flag_propagation
        self.mid_channels = mid_channels
        # if self.flag_propagation:
        #     self.conv10_ab = nn.Conv2d(128, self.mid_channels, 1, 1)
        # else:
        #     self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        # add self.relux_x
        self.relu1_1 = nn.ReLU()
        self.relu1_2 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.relu3_1 = nn.ReLU()
        self.relu3_2 = nn.ReLU()
        self.relu3_3 = nn.ReLU()
        self.relu4_1 = nn.ReLU()
        self.relu4_2 = nn.ReLU()
        self.relu4_3 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu5_2 = nn.ReLU()
        self.relu5_3 = nn.ReLU()
        self.relu6_1 = nn.ReLU()
        self.relu6_2 = nn.ReLU()
        self.relu6_3 = nn.ReLU()
        self.relu7_1 = nn.ReLU()
        self.relu7_2 = nn.ReLU()
        self.relu7_3 = nn.ReLU()
        self.relu8_1_comb = nn.ReLU()
        self.relu8_2 = nn.ReLU()
        self.relu8_3 = nn.ReLU()
        self.relu9_1_comb = nn.ReLU()
        self.relu9_2 = nn.ReLU()
        self.relu10_1_comb = nn.ReLU()
        self.relu10_2 = nn.LeakyReLU(0.2, True)

        print("replace all deconv with [nearest + conv]")
        self.conv8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, 3, 1, 1))
        self.conv9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, 3, 1, 1))
        self.conv10_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, 1, 1))

        print("replace all batchnorm with instancenorm")
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv4_3norm = nn.InstanceNorm2d(512)
        self.conv5_3norm = nn.InstanceNorm2d(512)
        self.conv6_3norm = nn.InstanceNorm2d(512)
        self.conv7_3norm = nn.InstanceNorm2d(512)
        self.conv8_3norm = nn.InstanceNorm2d(256)
        self.conv9_2norm = nn.InstanceNorm2d(128)

    def forward(self, x):
        """ x: gray image (1 channel), ab(2 channel), ab_err, ba_err"""
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)
        conv2_1 = self.relu2_1(self.conv2_1(conv1_2norm_ss))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)
        conv3_1 = self.relu3_1(self.conv3_1(conv2_2norm_ss))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)
        conv4_1 = self.relu4_1(self.conv4_1(conv3_3norm_ss))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)
        conv5_1 = self.relu5_1(self.conv5_1(conv4_3norm))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)
        conv6_1 = self.relu6_1(self.conv6_1(conv5_3norm))
        conv6_2 = self.relu6_2(self.conv6_2(conv6_1))
        conv6_3 = self.relu6_3(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)
        conv7_1 = self.relu7_1(self.conv7_1(conv6_3norm))
        conv7_2 = self.relu7_2(self.conv7_2(conv7_1))
        conv7_3 = self.relu7_3(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)
        conv8_1_comb = self.relu8_1_comb(conv8_1 + conv3_3_short)
        conv8_2 = self.relu8_2(self.conv8_2(conv8_1_comb))
        conv8_3 = self.relu8_3(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)
        conv9_1_comb = self.relu9_1_comb(conv9_1 + conv2_2_short)
        conv9_2 = self.relu9_2(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)
        conv10_1_comb = self.relu10_1_comb(conv10_1 + conv1_2_short)
        conv10_2 = self.relu10_2(self.conv10_2(conv10_1_comb))
        conv10_ab = self.conv10_ab(conv10_2)

        if self.flag_propagation:
            return conv10_ab
        else:
            return torch.tanh(conv10_ab) * 128


# 00810 v3—1: Add coarse to fine:
# resolution: 224x384 112x192 56x96
class ColorVidNet_wBasicVSR_v3(nn.Module):
    def __init__(self, ic, flag_propagation = False, mid_channels = 64):
        super(ColorVidNet_wBasicVSR_v3, self).__init__()
        self.size64 = ColorVidNet_wBasicVSR_v2(ic, flag_propagation = False, mid_channels = 64)
        self.size128 = ColorVidNet_wBasicVSR_v2(ic + 2, flag_propagation = False, mid_channels = 64)
        self.size256 = ColorVidNet_wBasicVSR_v2(ic + 2, flag_propagation = False, mid_channels = 64)

        # self.upconv1 = nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1, 
        #                                                output_padding=1, groups=1, bias=True, dilation=1)
        # self.upconv2 = nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1, 
        #                                                output_padding=1, groups=1, bias=True, dilation=1)
        self.upconv1 = nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1, 
                                                       output_padding=1, groups=1, bias=True, dilation=1)
        self.upconv2 = nn.ConvTranspose2d(2, 2, 3, stride=2, padding=1, 
                                                       output_padding=1, groups=1, bias=True, dilation=1)

    def forward(self, x):

        input_pyr_b1 = x[0]  # 224
        input_pyr_b2 = x[1]  # 112
        input_pyr_b3 = x[2]  # 56
        # print(input_pyr_b1.shape, input_pyr_b2.shape, input_pyr_b3.shape)

        # first layer res64
        out_size64 = self.size64(input_pyr_b3)

        # second layer res128
        up_out_size64 = self.upconv1(out_size64)
        input_size128 = torch.cat([input_pyr_b2, up_out_size64], dim=1)
        out_size128 = self.size128(input_size128)

        # print(up_out_size64.shape, out_size64.shape, input_size128.shape);assert 1==0

        # third layer res256
        up_out_size128 = self.upconv2(out_size128)
        input_size256 = torch.cat([input_pyr_b1, up_out_size128], dim=1)
        out_size256 = self.size128(input_size256)

        result = [out_size256, out_size128, out_size64]
        return result


class ResidualBlocksWithInputConv_adaptive_for_colorization(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        main.append(nn.Conv2d(out_channels, 2, 1, 1))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """

        output = self.main(feat)
        return torch.tanh(output) * 128


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017

    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=True, logger=logger)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, '
                            f'but got {type(pretrained)}.')

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.

        Note that in this function, the images are already resized to a
        multiple of 32.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, c, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(
                F.avg_pool2d(
                    input=ref[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.append(
                F.avg_pool2d(
                    input=supp[-1],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(
                    input=flow,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([
                    ref[level],
                    flow_warp(
                        supp[level],
                        flow_up.permute(0, 2, 3, 1),
                        padding_mode='border'), flow_up
                ], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_up, w_up),
            mode='bilinear',
            align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(
            input=self.compute_flow(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            ConvModule(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3,
                norm_cfg=None,
                act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)

