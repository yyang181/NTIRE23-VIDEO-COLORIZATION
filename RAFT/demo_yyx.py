import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)



DEVICE = 'cuda'

def exists_or_mkdir(path, verbose=False):
    try:
        if not os.path.exists(path):
            if verbose:
                print("creates %s ..."%path)  
            os.makedirs(path)
            return False
        else:
            if verbose:
                print("%s exists ..."%path)  
            return True     
    except Exception as e:
         print(e)

def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, save_path):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(save_path, img_flo[:, :, [2,1,0]])
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

def vis_yyx(img, img2, flo, save_path):
    # img = img[0].permute(1,2,0).cpu().numpy()
    # flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # # map flow to rgb image
    # flo = flow_viz.flow_to_image(flo)
    # img_flo = np.concatenate([img, flo], axis=0)

    # cv2.imwrite(save_path, img_flo[:, :, [2,1,0]])

    wimg2 = flow_warp(img, flo.permute(0, 2, 3, 1), padding_mode='border')
    wimg2 = wimg2[0].permute(1,2,0).cpu().numpy()
    cv2.imwrite(save_path.replace('.png', '_warped.png'), wimg2[:, :, [2,1,0]])

    img2 = img2[0].permute(1,2,0).cpu().numpy()
    cv2.imwrite(save_path, img2)

def vis_yyx_all(img, img2, flo, save_path):
    # warped image2
    wimg2 = flow_warp(img, flo.permute(0, 2, 3, 1), padding_mode='border')
    wimg2 = wimg2[0].permute(1,2,0).cpu().numpy()
    cv2.imwrite(save_path.replace('.png', '_warped.png'), wimg2[:, :, [2,1,0]])

    # image2 
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    cv2.imwrite(save_path, img2)

    # image 2 and flow 
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img2, flo], axis=0)
    cv2.imwrite(save_path.replace('.png', '_ConcatWithFlo.png'), img_flo[:, :, [2,1,0]])

def show_warp(img, img2, flo, save_path, imgname1, imgname2):
    wimg2 = flow_warp(img, flo.permute(0, 2, 3, 1), padding_mode='border')
    wimg2 = wimg2[0].permute(1,2,0).cpu().numpy()
    cv2.imwrite(save_path.replace('.png', '_WarpedFrom_%s.png'%imgname1), wimg2[:, :, [2,1,0]])

    img2 = img2[0].permute(1,2,0).cpu().numpy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img2)

def show_warp_wref(img, img2, ref, flo, save_path, imgname1, imgname2):
    wimg2 = flow_warp(ref, flo.permute(0, 2, 3, 1), padding_mode='border')
    wimg2 = wimg2[0].permute(1,2,0).cpu().numpy()
    cv2.imwrite(save_path.replace('.png', '_WarpedFrom_%s.png'%imgname1), wimg2[:, :, [2,1,0]])

    # img2 = img2[0].permute(1,2,0).cpu().numpy()
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, img2)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        idx = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            idx += 1
            # print(imfile1, imfile2)
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            ref = load_image(imfile1.replace(args.path, args.ref_path)) if args.ref_path != None else None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)

            method_of_vis = 'show_warp_wref' # 'vis' or 'vis_yyx' or 'vis_yyx_all' or 'show_warp' or 'show_warp_wref'
            
            '''
            OUTPUT:
            vis: concat input with its flow
            show_warp: img2.png, img2warpedfromimg1.png
            '''
            if method_of_vis == 'show_warp' or method_of_vis == 'show_warp_wref':
                output_dir = args.output_path
                exists_or_mkdir(output_dir)
                output_imgname = os.path.basename(imfile2)
                save_path = os.path.join(output_dir, output_imgname)
                print(idx, save_path)
            else:
                output_dir = args.output_path
                exists_or_mkdir(output_dir)
                output_imgname = os.path.basename(imfile1)
                save_path = os.path.join(output_dir, output_imgname)
                print(idx, save_path)
            if method_of_vis == 'vis':
                vis(image1, flow_up, save_path)
            elif method_of_vis == 'vis_yyx':
                vis_yyx(image1, image2, flow_up, save_path)
            elif method_of_vis == 'vis_yyx_all':
                vis_yyx_all(image1, image2, flow_up, save_path)
            elif method_of_vis == 'show_warp':
                show_warp(image1, image2, flow_up, save_path, os.path.basename(imfile1).split('.')[0], os.path.basename(imfile2))
            elif method_of_vis == 'show_warp_wref':
                assert args.ref_path != None
                show_warp_wref(image1, image2, ref, flow_up, save_path, os.path.basename(imfile1).split('.')[0], os.path.basename(imfile2))

def raft_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="dataset for evaluation")
    parser.add_argument('--ref_path', default=None, help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="dataset for evaluation")
    parser.add_argument('--ref_path', default=None, help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
