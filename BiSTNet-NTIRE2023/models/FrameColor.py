import torch
from utils.util import *

def warp_color_wBasicVSR(IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise=0, temperature=0.01):
    IA_rgb_from_gray = torch.zeros_like(IA_l).repeat(1,1,3,1,1)
    for i in range(IA_l.size(1)):
        IA_rgb_from_gray[:, i, :, :, :] = gray2rgb_batch(IA_l[:, i, :, :, :])

    # print(IA_rgb_from_gray.shape)
    # assert 1==0
    nonlocal_BA_lab_list = []
    similarity_map_list = []
    B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B
    for i in range(IA_l.size(1)):
        with torch.no_grad():
            A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
                IA_rgb_from_gray[:, i, :, :, :], ["r12", "r22", "r32", "r42", "r52"], preprocess=True
            )

        # NOTE: output the feature before normalization
        features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]

        A_relu2_1 = feature_normalize(A_relu2_1)
        A_relu3_1 = feature_normalize(A_relu3_1)
        A_relu4_1 = feature_normalize(A_relu4_1)
        A_relu5_1 = feature_normalize(A_relu5_1)
        B_relu2_1 = feature_normalize(B_relu2_1)
        B_relu3_1 = feature_normalize(B_relu3_1)
        B_relu4_1 = feature_normalize(B_relu4_1)
        B_relu5_1 = feature_normalize(B_relu5_1)

        nonlocal_BA_lab_single, similarity_map_single = nonlocal_net(
            IB_lab,
            A_relu2_1,
            A_relu3_1,
            A_relu4_1,
            A_relu5_1,
            B_relu2_1,
            B_relu3_1,
            B_relu4_1,
            B_relu5_1,
            temperature=temperature,
        )

        nonlocal_BA_lab_list.append(nonlocal_BA_lab_single)
        similarity_map_list.append(similarity_map_single)
    nonlocal_BA_lab = torch.cat(nonlocal_BA_lab_list, dim=0)
    similarity_map = torch.cat(similarity_map_list, dim=0)

    # print(nonlocal_BA_lab_single.shape, nonlocal_BA_lab.shape)
    # print(similarity_map_single.shape, similarity_map.shape)
    # assert 1==0
    return nonlocal_BA_lab, similarity_map, features_A


def warp_color(IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise=0, temperature=0.01):
    IA_rgb_from_gray = gray2rgb_batch(IA_l)
    with torch.no_grad():
        A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
            IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

    # NOTE: output the feature before normalization
    features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]

    A_relu2_1 = feature_normalize(A_relu2_1)
    A_relu3_1 = feature_normalize(A_relu3_1)
    A_relu4_1 = feature_normalize(A_relu4_1)
    A_relu5_1 = feature_normalize(A_relu5_1)
    B_relu2_1 = feature_normalize(B_relu2_1)
    B_relu3_1 = feature_normalize(B_relu3_1)
    B_relu4_1 = feature_normalize(B_relu4_1)
    B_relu5_1 = feature_normalize(B_relu5_1)

    nonlocal_BA_lab, similarity_map = nonlocal_net(
        IB_lab,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=temperature,
    )

    return nonlocal_BA_lab, similarity_map, features_A

def warp_color_v0_baseline_double(IA_l, IB_lab1, IB_lab2, features_B1, features_B2, vggnet, nonlocal_net, colornet, feature_noise=0, temperature=0.01):
    IA_rgb_from_gray = gray2rgb_batch(IA_l)
    with torch.no_grad():
        A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
            IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        B1_relu1_1, B1_relu2_1, B1_relu3_1, B1_relu4_1, B1_relu5_1 = features_B1
        B2_relu1_1, B2_relu2_1, B2_relu3_1, B2_relu4_1, B2_relu5_1 = features_B2

    # NOTE: output the feature before normalization
    features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]

    A_relu2_1 = feature_normalize(A_relu2_1)
    A_relu3_1 = feature_normalize(A_relu3_1)
    A_relu4_1 = feature_normalize(A_relu4_1)
    A_relu5_1 = feature_normalize(A_relu5_1)
    B1_relu2_1 = feature_normalize(B1_relu2_1)
    B1_relu3_1 = feature_normalize(B1_relu3_1)
    B1_relu4_1 = feature_normalize(B1_relu4_1)
    B1_relu5_1 = feature_normalize(B1_relu5_1)
    B2_relu2_1 = feature_normalize(B2_relu2_1)
    B2_relu3_1 = feature_normalize(B2_relu3_1)
    B2_relu4_1 = feature_normalize(B2_relu4_1)
    B2_relu5_1 = feature_normalize(B2_relu5_1)

    nonlocal_BA_lab, similarity_map = nonlocal_net(
        IB_lab1,
        IB_lab2,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B1_relu2_1,
        B1_relu3_1,
        B1_relu4_1,
        B1_relu5_1,
        B2_relu2_1,
        B2_relu3_1,
        B2_relu4_1,
        B2_relu5_1,
        temperature=temperature,
    )

    return nonlocal_BA_lab, similarity_map, features_A

def frame_colorization_wBasicVSR(
    IA_lab,
    IB_lab,
    IA_last_lab,
    features_B,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, :, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.autograd.set_grad_enabled(joint_training):
        '''
        Args:
            IA_lab (tensor): Current frame image in lab-channal with shape (n, t, c, h, w), torch.Size([1, 26, 3, 64, 64]) etc
            nonlocal_BA_lab (tensor): warped ab-channal colors w/L-channal with shape (t, c, h, w), torch.Size([26, 3, 64, 64]) etc
            similarity_map (tensor): confidence map with shape (t, 1, h, w), torch.Size([26, 1, 64, 64]) etc
            color_input (tensor): ColorVid input with shape (n, t, 4, h, w), torch.Size([1, 26, 4, 64, 64]) etc
            IA_ab_predict (tensor): ColorVid output with shape (n, t, 2, h, w), torch.Size([1, 26, 2, 64, 64]) etc
        '''
        nonlocal_BA_lab, similarity_map, features_A_gray = warp_color_wBasicVSR(
            IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]

        color_input = torch.cat((IA_l[0], nonlocal_BA_ab, similarity_map), dim=1).unsqueeze(0)
        # color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)

        IA_ab_predict = colornet(color_input)

        # print(IA_l.shape, nonlocal_BA_ab.shape, similarity_map.shape, color_input.shape, IA_ab_predict.shape)
        # assert 1==0

    return IA_ab_predict, nonlocal_BA_lab, features_A_gray


def frame_colorization(
    IA_lab,
    IB_lab,
    IA_last_lab,
    features_B,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.autograd.set_grad_enabled(joint_training):
        nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(
            IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
        color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)
        IA_ab_predict = colornet(color_input)

    return IA_ab_predict, nonlocal_BA_lab, features_A_gray

def frame_colorization_20230311_tcvc(
    IA_lab,
    IB_lab,
    IA_last_lab,
    features_B,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.no_grad():
        nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(
            IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]

    return nonlocal_BA_lab, similarity_map

def frame_colorization_0618_tcvc(
    IA_lab,
    IB_lab,
    IA_last_lab,
    features_B,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.no_grad():
        nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(
            IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
        color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)

    with torch.autograd.set_grad_enabled(joint_training):
        IA_ab_predict = colornet(color_input)

    return IA_ab_predict, nonlocal_BA_lab, features_A_gray, similarity_map

def frame_colorization_0618_tcvc_v0_baseline_double(
    IA_lab,
    IB_lab1,
    IB_lab2,
    IA_last_lab,
    features_B1,
    features_B2,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.no_grad():
        nonlocal_BA_lab, similarity_map, features_A_gray = warp_color_v0_baseline_double(
            IA_l, IB_lab1, IB_lab2, features_B1, features_B2, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
        color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)

    with torch.autograd.set_grad_enabled(joint_training):
        IA_ab_predict = colornet(color_input)

    return IA_ab_predict, nonlocal_BA_lab, features_A_gray, similarity_map


# def frame_colorization_0618_tcvc(
#     IA_lab,
#     IB_lab,
#     IA_last_lab,
#     features_B,
#     vggnet,
#     nonlocal_net,
#     colornet,
#     joint_training=True,
#     feature_noise=0,
#     luminance_noise=0,
#     temperature=0.01,
# ):

#     IA_l = IA_lab[:, 0:1, :, :]
#     if luminance_noise:
#         IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

#     with torch.autograd.set_grad_enabled(joint_training):
#         nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(
#             IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
#         )
#         nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
#         color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)
#         IA_ab_predict = colornet(color_input)

#     return IA_ab_predict, nonlocal_BA_lab, features_A_gray, similarity_map