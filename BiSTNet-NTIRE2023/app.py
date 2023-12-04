from __future__ import print_function
import sys
import os

sys.path.append('BiSTNet-NTIRE2023')
os.system("pip uninstall torch -y")
os.system("pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1")
os.system("mim install mmcv-full")
os.system("git clone -b 0.x https://github.com/open-mmlab/mmediting.git")
os.system("pip3 install -e ./mmediting/")

os.system("wget https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/releases/download/v1.0.3/checkpoints.zip")
os.system("wget https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/releases/download/v1.0.3/data.zip")
os.system("wget https://github.com/yyang181/NTIRE23-VIDEO-COLORIZATION/releases/download/v1.0.3/models.zip")
os.system("unzip checkpoints.zip && data.zip && models.zip")

from moviepy.editor import *
import gradio as gr
import argparse
import glob
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
import PIL
from PIL import Image
from tqdm import tqdm

import lib.TestTransforms as transforms
from models.ColorVidNet import SPyNet

from models.FrameColor import frame_colorization_20230311_tcvc as frame_colorization

from models.NonlocalNet import VGG19_pytorch, WarpNet_debug
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, save_frames_wOriName, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

from collections import OrderedDict

from torchvision import utils as vutils
from utils.util import gray2rgb_batch
import cv2

# PSNR SSIM
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

# mmedit flow_warp
from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN,
                                  flow_warp, make_layer)


# ATB block
from models.ColorVidNet import ColorVidNet_wBasicVSR_v2 as ColorVidNet
from models.ColorVidNet import ColorVidNet_wBasicVSR_v3
from models.ColorVidNet import ATB_block as ATB

# RAFT
from models.raft_core.raft import RAFT 


# SuperSloMo
import models.superslomo_model as Superslomo
from torchvision import transforms as superslomo_transforms
from torch.functional import F
from collections import OrderedDict


# HED
from models.hed import Network as Hed


# Proto Seg
import pickle
from models.protoseg_core.segmentor.tester import Tester_inference as Tester
from models.protoseg_core.lib.utils.tools.logger import Logger as Log
from models.protoseg_core.lib.utils.tools.configer import Configer
from PIL import Image
from models.protoseg_core.lib.vis.palette import get_cityscapes_colors, get_ade_colors, get_lip_colors, get_camvid_colors
from models.protoseg_core.lib.utils.helpers.file_helper import FileHelper
from models.protoseg_core.lib.utils.helpers.image_helper import ImageHelper

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


def tensor_gray2rgb(input):
    gray_IA = input
    ab_channal = torch.cat([torch.zeros_like(input), torch.zeros_like(input)], dim=1)
    gray_IA_rgb_from_gray = batch_lab2rgb_transpose_mc(gray_IA, ab_channal)
    return gray_IA_rgb_from_gray

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

def save_image_tensor(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def save_image_tensor_lab(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)

def ColorVid_inference(I_list, I1reference_video, features_B, vggnet, nonlocal_net, colornet, joint_training=False, flag_forward=True):
    # ref1 
    I_last_lab_predict = None
    colorvid1 = []
    similarity_map_list = []
    I_reference_lab = I1reference_video

    iter_item = range(len(I_list)) if flag_forward else range(len(I_list)-1, -1, -1)
    print('ColorVid_inference1') if flag_forward else print('ColorVid_inference2') 
    for index, i_idx in enumerate(tqdm(iter_item)):
    # for i_idx in iter_item:
        with torch.autograd.set_grad_enabled(joint_training):
            I_current_lab = I_list[i_idx]
            if I_last_lab_predict is None:
                I_last_lab_predict = torch.zeros_like(I_current_lab).cuda()
            
            I_current_nonlocal_lab_predict, similarity_map = frame_colorization(
                I_current_lab,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                joint_training=joint_training,
                feature_noise=0,
                temperature=1e-10,
            )
            # I_last_lab_predict = torch.cat((I_current_lab[:,:1,:,:], I_current_ab_predict), dim=1)
            colorvid1.append(I_current_nonlocal_lab_predict)
            similarity_map_list.append(similarity_map)

    return colorvid1, similarity_map_list

def compute_flow(lrs, raft, flag_save_flow_warp):
    n, t, c, h, w = lrs.size()
    flows_forward = []
    flows_backward = []
    with torch.no_grad():
        idx = 0
        for image1, image2 in zip(lrs[0,:-1,:,:,:], lrs[0,1:,:,:,:]):
            image1 = image1.unsqueeze(0) * 255.
            image2 = image2.unsqueeze(0) * 255.            

            flow_low, flow_forward = raft(image2, image1, iters=20, test_mode=True)
            flow_low, flow_backward = raft(image1, image2, iters=20, test_mode=True)
            flows_forward.append(flow_forward)
            flows_backward.append(flow_backward)
    return flows_forward, flows_backward


def bipropagation(colorvid1, colorvid2, I_list, flownet, atb, flag_save_flow_warp):
    I_gray2rgbbatach_list = [gray2rgb_batch(I[:,:1,:,:]).unsqueeze(0) for I in I_list]

    lrs = torch.cat(I_gray2rgbbatach_list, dim = 1)
    n, t, c, h, w = lrs.size()
    flows_forward, flows_backward = compute_flow(lrs, flownet, flag_save_flow_warp)

    # return fused
    return flows_forward, flows_backward

def HED_EdgeMask(I_list):
    joint_training = False
    I_current_l = torch.cat(I_list, dim = 0)[:,:1,:,:]
    I_current_lll = torch.cat([I_current_l, I_current_l, I_current_l], dim=1)

    ###### HED: Edge Detection ######
    tenInput2 = I_current_lll

    with torch.autograd.set_grad_enabled(joint_training):
        hed_edge2 = hed(tenInput2).clip(0.0, 1.0)

    hed_edge_ori2 = hed_edge2
    return hed_edge_ori2

def proto_segmask(I_list, flag_save_protoseg=False):
    # trans input resolution
    I_current_l = torch.cat(I_list, dim = 0)[:,:1,:,:]
    I_current_lll = torch.cat([I_current_l, I_current_l, I_current_l], dim=1)
    input_protoseg = trans_forward_protoseg_lll(I_current_lll)

    configer = Configer()
    data_dir = configer.get('data', 'data_dir')
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    abs_data_dir = [os.path.expanduser(x) for x in data_dir]
    configer.update(['data', 'data_dir'], abs_data_dir)
    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)
    if configer.get('logging', 'log_to_file'):
        log_file = configer.get('logging', 'log_file')
        new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file)
    else:
        configer.update(['logging', 'logfile_level'], None)
    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    model = Tester(configer)

    with torch.no_grad():
        outputs = model.test_deep_exemplar(input_protoseg)
    return outputs


def colorize_video(opt, input_path, reference_file, output_path, nonlocal_net, colornet, fusenet, vggnet, flownet, flag_lf_split_test_set, start_idx, end_idx):
    # parameters for wls filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    # processing folders
    mkdir_if_not(output_path)
    files = glob.glob(output_path + "*")
    print("processing the folder:", input_path)
    path, dirs, filenames = os.walk(input_path).__next__()
    file_count = len(filenames)
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))
    
    if flag_lf_split_test_set:
        filenames = filenames[start_idx:end_idx]
        print('num of testing images: %s starts from: %s ends from: %s'%(len(filenames), filenames[0], filenames[-1]))

    transform = transforms.Compose(
        # [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
        [superslomo_transforms.Resize(opt_image_size), RGB2Lab(), ToTensor(), Normalize()]
    )

    transform_full_l = transforms.Compose(
        # [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
        [RGB2Lab(), ToTensor(), Normalize()]
    )

    I_list = [Image.open(os.path.join(input_path, frame_name)).convert('RGB') for frame_name in filenames]
    I_list_large = [transform(frame1).unsqueeze(0).cuda() for frame1 in I_list]

    I_list_large_full_l = [transform_full_l(frame1).unsqueeze(0).cuda() for frame1 in I_list]

    I_list = [torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear") for IA_lab_large in I_list_large]

    print("reference name1:", reference_file[start_idx])
    ref_name1 = reference_file[start_idx]
    with torch.no_grad():
        frame_ref = Image.open(ref_name1).convert('RGB')
        IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
        IB_lab1 = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
        I_reference_rgb_from_gray = gray2rgb_batch(IB_lab1[:, 0:1, :, :])
        features_B1 = vggnet(I_reference_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    print("reference name2:", reference_file[end_idx-1])
    ref_name2 = reference_file[end_idx-1]
    with torch.no_grad():
        frame_ref = Image.open(ref_name2).convert('RGB')
        IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
        IB_lab2 = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")
        I_reference_rgb_from_gray = gray2rgb_batch(IB_lab2[:, 0:1, :, :])
        features_B2 = vggnet(I_reference_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)


    # ColorVid inference
    colorvid1, similarity_map_list1 = ColorVid_inference(I_list, IB_lab1, features_B1, vggnet, nonlocal_net, colornet, joint_training=False, flag_forward=True)
    colorvid2, similarity_map_list2 = ColorVid_inference(I_list, IB_lab2, features_B2, vggnet, nonlocal_net, colornet, joint_training=False, flag_forward=False)
    colorvid2.reverse()
    similarity_map_list2.reverse()

    # FUSION SimilarityMap
    similarityMap = []
    for i in range(len(similarity_map_list1)):
        # Fusion Mask Test
        FusionMask = torch.gt(similarity_map_list1[i], similarity_map_list2[i])
        FusionMask = torch.cat([FusionMask,FusionMask,FusionMask], dim = 1)

        Fused_Color = colorvid2[i]
        Fused_Color[FusionMask] = colorvid1[i][FusionMask]
        similarityMap.append(Fused_Color)

    # HED EdgeMask
    edgemask = HED_EdgeMask(I_list)

    # Proto Seg
    segmask = proto_segmask(I_list, flag_save_protoseg=False)

    flows_forward, flows_backward = bipropagation(colorvid1, colorvid2, I_list, flownet, atb, flag_save_flow_warp=False)

    print('fusenet v1: concat ref1+ref2')
    joint_training = False
    for index, i_idx in enumerate(tqdm(range(len(I_list)))):
        I_current_l = I_list[i_idx][:,:1,:,:]
        I_current_ab = I_list[i_idx][:,1:,:,:]

        # module: atb_test
        feat_fused, ab_fuse_videointerp, ab_fuse_atb = atb(colorvid1, colorvid2, flows_forward, flows_backward)

        fuse_input = torch.cat([I_list[i_idx][:,:1,:,:], colorvid1[i_idx][:,1:,:,:], colorvid2[i_idx][:,1:,:,:], feat_fused[i_idx], segmask[i_idx,:,:,:].unsqueeze(0), edgemask[i_idx,:,:,:].unsqueeze(0), similarityMap[i_idx][:,1:,:,:]], dim=1)
        
        with torch.no_grad():
            level1_shape = [fuse_input.shape[2], fuse_input.shape[3]]
            level2_shape = [int(fuse_input.shape[2]/2), int(fuse_input.shape[3]/2)]
            level3_shape = [int(fuse_input.shape[2]/4), int(fuse_input.shape[3]/4)]

            # v0 
            resize_b1tob2 = transform_lib.Resize(level2_shape)
            resize_b2tob3 = transform_lib.Resize(level3_shape)

            input_pyr_b1 = fuse_input
            input_pyr_b2 = resize_b1tob2(fuse_input)
            input_pyr_b3 = resize_b2tob3(input_pyr_b2)


            input_fusenet = [input_pyr_b1, input_pyr_b2, input_pyr_b3]
            output_fusenet = fusenet(input_fusenet)

            I_current_ab_predict = output_fusenet[0]


        IA_lab_large = I_list_large_full_l[i_idx]
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
        )
        curr_predict = (
            torch.nn.functional.interpolate(curr_predict, size=opt_image_size_ori, mode="bilinear")
        )


        # filtering
        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        save_frames_wOriName(IA_predict_rgb, output_path, image_name=filenames[index])

def load_pth(model, pth_path):
    nonlocal_test_path = pth_path
    state_dict_nonlocal_net = torch.load(nonlocal_test_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict_nonlocal_net.items():
        param = k.split(".")
        k = ".".join(param[1:])
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)



pretrain_model_url = {
	'safmn_x2': 'https://github.com/sunny2109/SAFMN/releases/download/v0.1.0/SAFMN_L_Real_LSDIR_x2.pth',
	'safmn_x4': 'https://github.com/sunny2109/SAFMN/releases/download/v0.1.0/SAFMN_L_Real_LSDIR_x4-v2.pth',
}


# # download weights
# if not os.path.exists('./experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2.pth'):
# 	load_file_from_url(url=pretrain_model_url['safmn_x2'], model_dir='./experiments/pretrained_models/', progress=True, file_name=None)

# if not os.path.exists('./experiments/pretrained_models/SAFMN_L_Real_LSDIR_x4-v2.pth'):
# 	load_file_from_url(url=pretrain_model_url['safmn_x4'], model_dir='./experiments/pretrained_models/', progress=True, file_name=None)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def img2patch(lq, scale=4, crop_size=512):
    b, c, hl, wl = lq.size()    
    h, w = hl*scale, wl*scale
    sr_size = (b, c, h, w)
    assert b == 1

    crop_size_h, crop_size_w = crop_size // scale * scale, crop_size // scale * scale

    #adaptive step_i, step_j
    num_row = (h - 1) // crop_size_h + 1
    num_col = (w - 1) // crop_size_w + 1

    import math
    step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
    step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

    step_i = step_i // scale * scale
    step_j = step_j // scale * scale

    parts = []
    idxes = []

    i = 0  # 0~h-1
    last_i = False
    while i < h and not last_i:
        j = 0
        if i + crop_size_h >= h:
            i = h - crop_size_h
            last_i = True

        last_j = False
        while j < w and not last_j:
            if j + crop_size_w >= w:
                j = w - crop_size_w
                last_j = True
            parts.append(lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
            idxes.append({'i': i, 'j': j})
            j = j + step_j
        i = i + step_i

    return torch.cat(parts, dim=0), idxes, sr_size


def patch2img(outs, idxes, sr_size, scale=4, crop_size=512):
    preds = torch.zeros(sr_size).to(outs.device)
    b, c, h, w = sr_size

    count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
    crop_size_h, crop_size_w = crop_size // scale * scale, crop_size // scale * scale

    for cnt, each_idx in enumerate(idxes):
        i = each_idx['i']
        j = each_idx['j']
        preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += outs[cnt]
        count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

    return (preds / count_mt).to(outs.device)


os.makedirs('./results', exist_ok=True)

def video2frames(video_dir, out_frames_dir="None"):
    os.makedirs(out_frames_dir, exist_ok=True)
    video = VideoFileClip(video_dir)
    # audio = video.audio
    # audio.write_audiofile(out_frames_dir + ".mp3")

    vidcap = cv2.VideoCapture(video_dir)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
        success,image = vidcap.read()
        # if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        #     break    
        if image is None:
            print("Fps is {}".format(fps))
            return 0    
        if count % 100 == 0:
            print("Video to frames: {}/frame{:04d}.png    Image shape:" .format(out_frames_dir, count),    image.shape)
        cv2.imwrite("{}/f{:03d}.png".format(out_frames_dir, count), image)     # save frame as JPEG file
        count += 1
    vidcap.release()
    # audio.release()
    print("Fps is {}".format(fps))
    return int(fps)

def inference(video, ref, width, height, large_input_flag, color_fix):
    fps = video2frames(video, out_frames_dir="input")
    print("fps is {}".format(fps))

    epoch = 105000
    dirName_ckp = '20230311_NTIRE2023'
    nonlocal_test_path = os.path.join("checkpoints/", "finetune_test0610/nonlocal_net_iter_6000.pth")
    color_test_path = os.path.join("checkpoints/", "finetune_test0610/colornet_iter_6000.pth")
    fusenet_path = os.path.join("checkpoints/", "%s/fusenet_iter_%s.pth"%(dirName_ckp, epoch))
    atb_path = os.path.join("checkpoints/", "%s/atb_iter_%s.pth"%(dirName_ckp, epoch))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
    )
    parser.add_argument("--cuda", action="store_false")
   
    # 20230215 ntire test set 
    parser.add_argument("--clip_path", type=str, default="../demo_dataset/input", help="path of input clips")
    parser.add_argument("--ref_path", type=str, default="../demo_dataset/ref", help="path of refernce images")
    parser.add_argument("--output_path", type=str, default="../demo_dataset/output", help="path of output clips")

    start_idx = 0
    end_idx = -1

    # RAFT params
    parser.add_argument('--model', default='data/raft-sintel.pth', type=str, help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    opt = parser.parse_args()
    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
    cudnn.benchmark = True
    print("running on GPU", opt.gpu_ids)

    opt.image_size = [width,height]
    opt.clip_path = image 
    opt.ref_path = ref

    opt_clip_path = opt.clip_path
    opt_ref_path = opt.ref_path
    opt_output_path = opt.output_path

    nonlocal_net = WarpNet_debug(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    fusenet = ColorVidNet_wBasicVSR_v3(33, flag_propagation = False)


    ### Flownet: raft version  
    flownet = RAFT(opt)

    ### ATB
    atb = ATB()

    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    load_pth(nonlocal_net, nonlocal_test_path)
    load_pth(colornet, color_test_path)
    load_pth(fusenet, fusenet_path)
    load_pth(flownet, opt.model)
    load_pth(atb, atb_path)
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    print("succesfully load fusenet model: ", fusenet_path)
    print("succesfully load flownet model: ", 'raft')
    print("succesfully load atb model: ", atb_path)

    fusenet.eval()
    fusenet.cuda()
    flownet.eval()
    flownet.cuda()
    atb.eval()
    atb.cuda()
    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()

 
    opt_image_size = opt.image_size

    # HED
    hed = Hed().cuda().eval()
    w0, h0 = opt_image_size[0], opt_image_size[1]
    w, h = (w0 // 32) * 32, (h0 // 32) * 32
    # forward l
    intWidth = 480
    intHeight = 320
    meanlab = [-50, -50, -50]   # (A - mean) / std
    stdlab = [100, 100, 100]   # (A - mean) / std
    trans_forward_hed_lll = superslomo_transforms.Compose([superslomo_transforms.Normalize(mean=meanlab, std=stdlab), superslomo_transforms.Resize([intHeight, intWidth])])
    # backward
    trans_backward = superslomo_transforms.Compose([superslomo_transforms.Resize([w0,h0])])

    # proto seg
    meanlab_protoseg = [0.485, 0.485, 0.485]   # (A - mean) / std
    stdlab_protoseg = [0.229, 0.229, 0.229]   # (A - mean) / std
    trans_forward_protoseg_lll = superslomo_transforms.Compose([superslomo_transforms.Normalize(mean=meanlab, std=stdlab), superslomo_transforms.Normalize(mean=meanlab_protoseg, std=stdlab_protoseg)])


    # dataset preprocessing for batch testing
    clips = sorted(os.listdir(opt_clip_path))
    opt_clip_path_ori = opt_clip_path
    opt_ref_path_ori = opt_ref_path
    opt_output_path_ori = opt_output_path

    for idx_clip, clip in enumerate(clips):
        dirTestImageName = os.path.join(opt_clip_path_ori, sorted(os.listdir(opt_clip_path_ori))[idx_clip])
        TestImageName = os.path.join(opt_clip_path_ori, sorted(os.listdir(opt_clip_path_ori))[idx_clip], os.listdir(dirTestImageName)[0])
        test_img = Image.open(TestImageName).convert('RGB')
        opt_image_size_ori = np.shape(test_img)[:2]

        opt_image_size = opt.image_size

        dirName_input = os.path.join(opt_clip_path_ori, clip)
        dirName_ref = os.path.join(opt_ref_path_ori, clip)
        dirName_output = os.path.join(opt_output_path_ori, clip)

        opt_clip_path = dirName_input
        opt_ref_path = dirName_ref
        opt_output_path = dirName_output

        print(idx_clip, clip, opt_clip_path, opt_ref_path, opt_output_path)

        exists_or_mkdir(dirName_output)
        clip_name = opt_clip_path.split("/")[-1]
        refs = os.listdir(opt_ref_path)
        refs.sort()

        ref_name = refs[start_idx].split('.')[0] + '_' + refs[end_idx].split('.')[0]

        len_interval = 50
        flag_lf_split_test_set = True

        for i in range(0, len(refs), len_interval):
            if i != 0:
                sub_ref = refs[i-1:i + len_interval]
                ActStartIdx = i-1
                ActEndIdx = i + len_interval 
            else:
                sub_ref = refs[i:i + len_interval]
                ActStartIdx = i
                ActEndIdx = i + len_interval
            ActEndIdx = min(ActEndIdx, len(refs))

            print(i, 'startImg: %s endImg: %s, ActStartIdx: %s, ActEndIdx: %s'%(sub_ref[0], sub_ref[-1], ActStartIdx, ActEndIdx))

            colorize_video(
                opt,
                opt_clip_path,
                [os.path.join(opt_ref_path, name) for name in refs],
                # os.path.join(opt_output_path, clip_name + "_" + ref_name.split(".")[0]),
                os.path.join(opt_output_path),
                nonlocal_net,
                colornet,
                fusenet,
                vggnet,
                flownet,
                flag_lf_split_test_set,
                ActStartIdx,
                ActEndIdx,
            )


# def inference(image, upscale, large_input_flag, color_fix):
# 	upscale = int(upscale) # convert type to int
# 	if upscale > 4: 
# 		upscale = 4 
# 	if 0 < upscale < 3:
# 		upscale = 2

# 	model = set_safmn(upscale)

# 	img = cv2.imread(str(image), cv2.IMREAD_COLOR)
# 	print(f'input size: {img.shape}')

# 	# img2tensor
# 	img = img.astype(np.float32) / 255.
# 	img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
# 	img = img.unsqueeze(0).to(device)

# 	# inference
# 	if large_input_flag:
# 		patches, idx, size = img2patch(img, scale=upscale)
# 		with torch.no_grad():
# 			n = len(patches)
# 			outs = []
# 			m = 1
# 			i = 0
# 			while i < n:
# 				j = i + m
# 				if j >= n:
# 					j = n
# 				pred = output = model(patches[i:j])
# 				if isinstance(pred, list):
# 					pred = pred[-1]
# 				outs.append(pred.detach())
# 				i = j
# 			output = torch.cat(outs, dim=0)

# 		output = patch2img(output, idx, size, scale=upscale)
# 	else:
# 		with torch.no_grad():
# 			output = model(img)

# 	# color fix
# 	if color_fix:
# 		img = F.interpolate(img, scale_factor=upscale, mode='bilinear')
# 		output = wavelet_reconstruction(output, img)
# 	# tensor2img
# 	output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
# 	if output.ndim == 3:
# 		output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
# 	output = (output * 255.0).round().astype(np.uint8)

# 	# save restored img
# 	save_path = f'results/out.png'
# 	cv2.imwrite(save_path, output)

# 	output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# 	return output, save_path



title = "Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution"
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/sunny2109/SAFMN' target='_blank'><b>Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution (ICCV 2023)</b></a>.<br>
"""
article = r"""
If SAFMN is helpful, please help to ⭐ the <a href='https://github.com/sunny2109/SAFMN' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/sunny2109/SAFMN?style=social)](https://github.com/sunny2109/SAFMN)

---
📝 **Citation**

If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{sun2023safmn,
    title={Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution},
    author={Sun, Long and Dong, Jiangxin and Tang, Jinhui and Pan, Jinshan},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2023}
}
```

<center><img src='https://visitor-badge.laobi.icu/badge?page_id=sunny2109/SAFMN' alt='visitors'></center>
"""

# demo = gr.Interface(video_identity, 
#                     gr.Video(), 
#                     "playable_video", 
#                     examples=[
#                         os.path.join(os.path.dirname(__file__), 
#                                      "video/video_sample.mp4")], 
#                     cache_examples=True)

demo = gr.Interface(
    inference, [
        gr.inputs.Video(label="video", type='mp4'),
        # gr.inputs.Image(type="filepath", label="Input"),
        gr.inputs.Number(default=448, label="Image size width"),
        gr.inputs.Number(default=896, label="Image size height"),
		gr.inputs.Checkbox(default=False, label="Memory-efficient inference"),
        gr.inputs.Checkbox(default=False, label="Color correction"),
    ],
    examples=[os.path.join(os.path.dirname(__file__), 
                                     "video/video_sample.mp4")],
    outputs = [
        gr.outputs.Image(type="numpy", label="Output"),
        gr.outputs.File(label="Download the output")
    ],
    title=title,
    description=description,
    article=article,       
)

demo.queue(concurrency_count=2)
demo.launch()