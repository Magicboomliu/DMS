import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import sys
sys.path.append("..")
from losses.metirc import *



def L1Loss(input, target):
    return (input - target).abs().mean()


def loss_disp_unsupervised(img_left, img_right, disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped = warp_disp(img_right, -disp)

    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask

    loss = 0.15 * L1Loss(image_warped * valid_mask, img_left * valid_mask) + \
           0.85 * (valid_mask * (1 - ssim(img_left, image_warped)) / 2).mean()
    return loss



def L1Loss_without_mean(input, target):
    return (input - target).abs()

# left-left and right-right
def loss_disp_unsupervised_multi(img_left, 
                                 img_right, 
                                 img_left_left,
                                 img_right_right,
                                 disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_right_right,-2*disp)
    image_warped_left3 = warp_disp(img_left_left,disp)
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   
    loss_2 = 0.15 * L1Loss_without_mean(image_warped_left3*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left3)) / 2)
                       
    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)
    loss_2_abs = torch.sum(loss_2,dim=1,keepdim=True)


    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,loss_1_abs,loss_2_abs], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_2 = torch.where(min_tensor_idx == 2, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    
    mask_0 = mask_0.type_as(img_left)
    mask_1 = mask_1.type_as(img_left)
    mask_2 = mask_2.type_as(img_left)
    
    mask_0= mask_0.float()
    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    
    mask = mask_0 + mask_1 + mask_2
    loss = loss_0 * mask_0 + loss_1 * mask_1  + loss_2 * mask_2
    
    loss = loss.mean()
                   
    return loss


# left-left and right-right and center
def loss_disp_unsupervised_multi_plus_center(img_left, 
                                 img_right, 
                                 img_left_left,
                                 img_right_right,
                                 img_center,
                                 
                                 disp, valid_mask=None, mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_right_right,-2*disp)
    image_warped_left3 = warp_disp(img_left_left,disp)
    image_warped_left4 = warp_disp(img_center,-0.5*disp)
    
    
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   
    loss_2 = 0.15 * L1Loss_without_mean(image_warped_left3*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left3)) / 2)
                   
    loss_3 = 0.15 * L1Loss_without_mean(image_warped_left4*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left4)) / 2)        
    
                       
    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)
    loss_2_abs = torch.sum(loss_2,dim=1,keepdim=True)
    loss_3_abs = torch.sum(loss_3,dim=1,keepdim=True)


    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,loss_1_abs,loss_2_abs,loss_3_abs], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True), torch.tensor(False))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True), torch.tensor(False))
    mask_2 = torch.where(min_tensor_idx == 2, torch.tensor(True), torch.tensor(False))
    mask_3 = torch.where(min_tensor_idx == 3, torch.tensor(True), torch.tensor(False))
    
    mask_0= mask_0.float()
    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    mask_3 = mask_3.float()
    
    mask = mask_0 + mask_1 + mask_2 + mask_3 
    loss = loss_0 * mask_0 + loss_1 * mask_1  + loss_2 * mask_2 + loss_3 * mask_3
    
    loss = loss.mean()
                   
    return loss


# center images
def loss_disp_unsupervised_center(img_left, 
                                 img_right,
                                 img_center,
                                 disp, 
                                 valid_mask=None, 
                                 mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_center,-0.5*disp)
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                       
    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)

    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,loss_1_abs], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True), torch.tensor(False))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True), torch.tensor(False))

    
    mask_0= mask_0.float()
    mask_1 = mask_1.float()

    mask = mask_0 + mask_1 
    loss = loss_0 * mask_0 + loss_1 * mask_1 
    
    loss = loss.mean()
                   
    return loss


# center images
def loss_disp_unsupervised_centerAll(
                                 img_left, 
                                 img_right,
                                 img_one_third,
                                 img_center,
                                 img_two_third,
                                 disp, 
                                 valid_mask=None, 
                                 mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_center,-0.5*disp)
    image_warped_left3 = warp_disp(img_one_third,-(disp*1.0/3))
    image_warped_left4 = warp_disp(img_two_third,-(disp*1.0/3)*2)
    
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   
    loss_2 = 0.15 * L1Loss_without_mean(image_warped_left3*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left3)) / 2)        

    loss_3 = 0.15 * L1Loss_without_mean(image_warped_left4*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left4)) / 2)   
                   
                   
                       
    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)
    loss_2_abs = torch.sum(loss_2,dim=1,keepdim=True)
    loss_3_abs = torch.sum(loss_3,dim=1,keepdim=True)
    

    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,
                                             loss_1_abs,
                                             loss_2_abs,
                                             loss_3_abs], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_2 = torch.where(min_tensor_idx == 2, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_3 = torch.where(min_tensor_idx == 3, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    
    mask_0= mask_0.float()
    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    mask_3 = mask_3.float()

    mask = mask_0 + mask_1 + mask_2 + mask_3
    
    loss = loss_0 * mask_0 + loss_1 * mask_1 + loss_2 * mask_2 + loss_3 * mask_3
    
    loss = loss.mean()
                   
    return loss



def loss_disp_unsupervised_All_In_All(
                                 img_left, 
                                 img_right,
                                 img_one_third,
                                 img_center,
                                 img_two_third,
                                 img_left_left,
                                 img_right_right,
                                 disp, 
                                 valid_mask=None, 
                                 mask=None):
    b, _, h, w = img_left.shape
    image_warped_left = warp_disp(img_right, -disp)
    image_warped_left2 = warp_disp(img_center,-0.5*disp)
    image_warped_left3 = warp_disp(img_one_third,-(disp*1.0/3))
    image_warped_left4 = warp_disp(img_two_third,-(disp*1.0/3)*2)
    image_warped_left5 = warp_disp(img_right_right,-2*disp)
    image_warped_left6 = warp_disp(img_left_left,disp)
    
    
    
    valid_mask = torch.ones(b, 1, h, w).to(img_left.device) if valid_mask is None else valid_mask
    if mask is not None:
        valid_mask = valid_mask * mask
        
        
    loss_0 = 0.15 * L1Loss_without_mean(image_warped_left*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left)) / 2)
                   
    loss_1 = 0.15 * L1Loss_without_mean(image_warped_left2*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left2)) / 2)
                   
    loss_2 = 0.15 * L1Loss_without_mean(image_warped_left3*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left3)) / 2)        

    loss_3 = 0.15 * L1Loss_without_mean(image_warped_left4*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left4)) / 2) 
                   
    loss_4 = 0.15 * L1Loss_without_mean(image_warped_left5*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left5)) / 2)        

    loss_5 = 0.15 * L1Loss_without_mean(image_warped_left6*valid_mask,img_left * valid_mask) + \
                   0.85 * (valid_mask * (1 - ssim(img_left, image_warped_left6)) / 2)
    
                       
    loss_0_abs = torch.sum(loss_0,dim=1,keepdim=True)
    loss_1_abs = torch.sum(loss_1,dim=1,keepdim=True)
    loss_2_abs = torch.sum(loss_2,dim=1,keepdim=True)
    loss_3_abs = torch.sum(loss_3,dim=1,keepdim=True)
    
    loss_4_abs = torch.sum(loss_4,dim=1,keepdim=True)
    loss_5_abs = torch.sum(loss_5,dim=1,keepdim=True)
    

    min_tensor_idx = torch.argmin(torch.cat([loss_0_abs,
                                             loss_1_abs,
                                             loss_2_abs,
                                             loss_3_abs,
                                             loss_4_abs,
                                             loss_5_abs
                                             ], dim=1), dim=1,keepdim=True) #[B,1,H,W]
    
    mask_0 = torch.where(min_tensor_idx == 0, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_1 = torch.where(min_tensor_idx == 1, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_2 = torch.where(min_tensor_idx == 2, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_3 = torch.where(min_tensor_idx == 3, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_4 = torch.where(min_tensor_idx == 4, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    mask_5 = torch.where(min_tensor_idx == 5, torch.tensor(True).to(min_tensor_idx.device), torch.tensor(False).to(min_tensor_idx.device))
    

    mask_0= mask_0.float()
    mask_1 = mask_1.float()
    mask_2 = mask_2.float()
    mask_3 = mask_3.float()
    mask_4= mask_4.float()
    mask_5 = mask_5.float()
    
    mask = mask_0 + mask_1 + mask_2 + mask_3 + mask_4 + mask_5
    loss = loss_0 * mask_0 + loss_1 * mask_1 + loss_2 * mask_2 + loss_3 * mask_3 + loss_4 * mask_4 + loss_5 * mask_5
    
    loss = loss.mean()
                   
    return loss






def loss_disp_smoothness(disp, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
            ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss



def loss_pam_photometric(img_left, img_right, att, valid_mask, mask=None):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(img_left.device)

    for idx_scale in range(len(att)):
        scale = img_left.size()[2] // valid_mask[idx_scale][0].size()[2]
        b, c, h, w = valid_mask[idx_scale][0].size()

        att_right2left = att[idx_scale][0]                          # b * h * w * w
        att_left2right = att[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]                  # b * 1 * h * w
        valid_mask_right = valid_mask[idx_scale][1]

        if mask is not None:
            valid_mask_left = valid_mask_left * (nn.AvgPool2d(scale)(mask[0].float()) > 0).float()
            valid_mask_right = valid_mask_right * (nn.AvgPool2d(scale)(mask[1].float()) > 0).float()

        img_left_scale  = F.interpolate(img_left,  scale_factor=1/scale, mode='bilinear')
        img_right_scale = F.interpolate(img_right, scale_factor=1/scale, mode='bilinear')

        img_right_warp = torch.matmul(att_right2left, img_right_scale.permute(0, 2, 3, 1).contiguous())
        img_right_warp = img_right_warp.permute(0, 3, 1, 2)
        img_left_warp = torch.matmul(att_left2right,  img_left_scale.permute(0, 2, 3, 1).contiguous())
        img_left_warp = img_left_warp.permute(0, 3, 1, 2)

        loss_scale = L1Loss(img_left_scale * valid_mask_left, img_right_warp * valid_mask_left) + \
                     L1Loss(img_right_scale * valid_mask_right, img_left_warp * valid_mask_right)

        loss = loss + weight[idx_scale] * loss_scale

    return loss



def loss_pam_cycle(att_cycle, valid_mask):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att_cycle[0][0].device)

    for idx_scale in range(len(att_cycle)):
        b, c, h, w = valid_mask[idx_scale][0].shape
        I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_cycle[0][0].device)

        att_left2right2left = att_cycle[idx_scale][0]
        att_right2left2right = att_cycle[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]
        valid_mask_right = valid_mask[idx_scale][1]

        loss_scale = L1Loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1), I * valid_mask_left.permute(0, 2, 3, 1)) + \
                     L1Loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1), I * valid_mask_right.permute(0, 2, 3, 1))

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_smoothness(att):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att[0][0].device)

    for idx_scale in range(len(att)):
        att_right2left = att[idx_scale][0]
        att_left2right = att[idx_scale][1]

        loss_scale = L1Loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
                     L1Loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
                     L1Loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
                     L1Loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

        loss = loss + weight[idx_scale] * loss_scale

    return loss

def warp_disp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

def ssim(img1, img2, window_size=11):
    _, channel, h, w = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)

def PAMStereoLossMultiScale(img_left,img_right,disp_pyramid,att,att_cycle,valid_mask,disp_gt):
    disp = disp_pyramid[-1]
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    weights = [0.8,1.0,1.2]
    
    total_loss_P = 0
    
    total_loss_S = 0
    
    for idx, d in enumerate(disp_pyramid):
        # loss-p
        loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)
    
        # loss-S
        loss_S = loss_disp_smoothness(disp, img_left)

        total_loss_P = total_loss_P + loss_P * weights[idx]
        
        total_loss_S = total_loss_S + loss_S * weights[idx]

    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = total_loss_P + 0.1 * total_loss_S + loss_PAM
    
    return loss, total_loss_P, total_loss_S, loss_PAM

def PAMStereoLoss(img_left,img_right,disp,att,att_cycle,valid_mask,disp_gt):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)

    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = loss_P + 0.1 * loss_S + loss_PAM
    
    return loss, loss_P, loss_S, loss_PAM


def PAMStereoLoss_Simple(img_left,img_right,disp,att,att_cycle,valid_mask,disp_gt):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, None, mask_left)
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)

    loss = loss_P + 0.1 * loss_S
    
    return loss, loss_P, loss_S,loss_S



def PAMLoss_OutSide(img_left,img_right,disp,
                    att,att_cycle,valid_mask,disp_gt,
                    img_left_left, img_right_right
                    ):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)
    loss_P_plus = loss_disp_unsupervised_multi(img_left=img_left,img_right=img_right,img_left_left=img_left_left,
                                                img_right_right=img_right_right,disp=disp)
    
    # loss_P = loss_P + loss_P_plus
    loss_P = loss_P_plus * 2.0
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)


    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = loss_P + 0.1 * loss_S + loss_PAM
    
    return loss, loss_P, loss_S, loss_PAM

def PAMLoss_OutSideCenter(img_left,img_right,disp,
                    att,att_cycle,valid_mask,disp_gt,
                    img_left_left, img_right_right,
                    img_center,
                    ):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)
    
    loss_P_plus = loss_disp_unsupervised_multi_plus_center(img_left=img_left,img_right=img_right,img_left_left=img_left_left,
                                                img_right_right=img_right_right,disp=disp,img_center=img_center)
    
    # loss_P_plus = loss_disp_unsupervised_multi(img_left=img_left,img_right=img_right,img_left_left=img_left_left,
    #                                             img_right_right=img_right_right,disp=disp)
    
    # loss_P = loss_P + loss_P_plus
    loss_P = loss_P_plus * 2.0
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)


    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = loss_P + 0.1 * loss_S + loss_PAM
    
    return loss, loss_P, loss_S, loss_PAM


def PAMLoss_Center(img_left,img_right,disp,
                    att,att_cycle,valid_mask,disp_gt,
                    img_center
                    ):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)
    loss_P_plus = loss_disp_unsupervised_center(img_left=img_left,
                                                img_right=img_right,
                                                img_center=img_center,
                                                disp=disp)
    
    # loss_P = loss_P + loss_P_plus
    loss_P = loss_P_plus * 2.0
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)


    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = loss_P + 0.1 * loss_S + loss_PAM
    
    return loss, loss_P, loss_S, loss_PAM


def PAMLoss_Center_All(img_left,
                       img_right,
                       disp,
                       att,
                       att_cycle,
                       valid_mask,
                       disp_gt,
                       img_center,
                       img_one_third,
                       img_two_one_third,
                    ):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, 
                                    F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)


    loss_P_plus =loss_disp_unsupervised_centerAll(img_left=img_left,
                                                  img_right=img_right,
                                                  img_one_third=img_one_third,
                                                  img_center=img_center,
                                                  img_two_third=img_two_one_third,
                                                  disp=disp)
    # loss_P = loss_P + loss_P_plus
    loss_P = loss_P_plus * 2.0
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)


    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = loss_P + 0.1 * loss_S + loss_PAM
    
    return loss, loss_P, loss_S, loss_PAM



def PAMLoss_Center_All_in_All(img_left,
                       img_right,
                       disp,
                       att,
                       att_cycle,
                       valid_mask,
                       disp_gt,
                       img_center,
                       img_one_third,
                       img_two_one_third,
                       img_left_left,
                       img_right_right
                    ):
    
    if disp_gt is not None:
        mask_left = ((disp_gt > 0) & (disp_gt < 192)).float()
    else:
        mask_left = torch.ones_like(disp)
    mask_right = torch.ones_like(mask_left)
    
    # loss-p
    loss_P = loss_disp_unsupervised(img_left, img_right, disp, 
                                    F.interpolate(valid_mask[-1][0], scale_factor=4, mode='nearest'), mask_left)


    loss_P_plus =loss_disp_unsupervised_All_In_All(img_left=img_left,
                                                   img_right=img_right,
                                                   img_one_third=img_one_third,
                                                   img_two_third=img_two_one_third,
                                                   img_left_left=img_left_left,
                                                   img_right_right=img_right_right,
                                                   disp=disp,
                                                   img_center=img_center)
    # loss_P = loss_P + loss_P_plus
    loss_P = loss_P_plus * 2.0
    
    # loss-S
    loss_S = loss_disp_smoothness(disp, img_left)


    # loss-PAM
    loss_PAM_P = loss_pam_photometric(img_left, img_right, att, valid_mask, [mask_left, mask_right])
    loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask)
    loss_PAM_S = loss_pam_smoothness(att)
    loss_PAM = loss_PAM_P + loss_PAM_S + loss_PAM_C

    loss = loss_P + 0.1 * loss_S + loss_PAM
    
    return loss, loss_P, loss_S, loss_PAM