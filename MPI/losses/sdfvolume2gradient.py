import torch
import torch.nn as nn
import torch.nn.functional as F


KITTI_RAW_WIDTH = 1242
KITTI_RAW_HEIGHT = 375
CROP_WIDTH = 960
CROP_HEIGHT = 320


def SDF2Graident(sdf_volume, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    '''
    SDF_volume: (B, D, H, W)
    K: (3, 3)
    hypo_depths: (D, )

    SDF_volume为1/8分辨率下
    注意K也要被scale到1/8分辨率
    '''
    batch_num, depth_num, height, width = sdf_volume.shape
    # i, j = torch.meshgrid(torch.linspace(0, width-1, width), torch.linspace(0, height-1, height), indexing='xy')
    # coords_2d = torch.stack([i, j, torch.ones_like(i)], dim=0)  # (3, H, W)
    # inv_K = torch.linalg.inv(K)
    # coords_3d = torch.einsum('ij,jhw->ihw', inv_K, coords_2d)  # (3, H, W)

    # hypo_depths = hypo_depths[None, :, None, None]  # (1, D, 1, 1)
    # coords_3d = coords_3d[:, None, ...] * hypo_depths  # (3, D, H, W)
    fx = fx_unit * width * (KITTI_RAW_WIDTH / CROP_WIDTH)
    fy = fy_unit * height * (KITTI_RAW_HEIGHT / CROP_HEIGHT)
    hypo_depths = torch.linspace(0, depth_num-1, depth_num).type_as(sdf_volume)
    hypo_depths[0] += 1e-4  # precision issue
    hypo_depths = fx * baseline / hypo_depths  # convert disparity to depth

    z_diffs = torch.abs(hypo_depths[:-1] - hypo_depths[1:])  # (D-1, )
    hypo_xs = 1 / fx * hypo_depths  # (D, )
    hypo_ys = 1 / fy * hypo_depths  # (D, )

    grad_x = torch.abs(sdf_volume[:, :, :, :-1] - sdf_volume[:, :, :, 1:]) / hypo_xs[None, :, None, None]  # (B, D, H, W-1)
    grad_y = torch.abs(sdf_volume[:, :, :-1, :] - sdf_volume[:, :, 1:, :]) / hypo_ys[None, :, None, None]  # (B, D, H-1, W)
    grad_z = torch.abs(sdf_volume[:, :-1, :, :] - sdf_volume[:, 1:, :, :]) / z_diffs[None, :, None, None]  # (B, D-1, H, W)

    sdf_gradient = torch.stack([grad_x[:, :depth_num-1, :height-1, :width-1],
                                grad_y[:, :depth_num-1, :height-1, :width-1],
                                grad_z[:, :depth_num-1, :height-1, :width-1]], dim=1)  # (B, 3, D-1, H-1, W-1)

    sdf_gradient_norm = torch.norm(sdf_gradient, dim=1)  # (B, D-1, H-1, W-1)

    return sdf_gradient_norm


def SDF2Gradient_multiscale(sdf_volume, center_disp, radius, fx_unit=0.58, fy_unit=1.92, baseline=0.54):
    '''
    SDF_volume: (B, D, H, W)
    K: (3, 3)
    hypo_depths: (D, )

    SDF_volume为1/8分辨率下
    注意K也要被scale到1/8分辨率
    '''
    batch_num, depth_num, height, width = sdf_volume.shape
    # i, j = torch.meshgrid(torch.linspace(0, width-1, width), torch.linspace(0, height-1, height), indexing='xy')
    # coords_2d = torch.stack([i, j, torch.ones_like(i)], dim=0)  # (3, H, W)
    # inv_K = torch.linalg.inv(K)
    # coords_3d = torch.einsum('ij,jhw->ihw', inv_K, coords_2d)  # (3, H, W)

    # hypo_depths = hypo_depths[None, :, None, None]  # (1, D, 1, 1)
    # coords_3d = coords_3d[:, None, ...] * hypo_depths  # (3, D, H, W)
    fx = fx_unit * width * (KITTI_RAW_WIDTH / CROP_WIDTH)
    fy = fy_unit * height * (KITTI_RAW_HEIGHT / CROP_HEIGHT)
    # hypo_depths = torch.linspace(0, depth_num-1, depth_num).type_as(sdf_volume)
    # 多尺度时，根据center_disp和radius决定hypo_depths
    hypo_depths = torch.linspace(center_disp-radius, center_disp+radius, 2*radius+1).type_as(sdf_volume)
    hypo_depths[0] += 1e-4  # precision issue
    hypo_depths = fx * baseline / hypo_depths  # convert disparity to depth

    z_diffs = torch.abs(hypo_depths[:-1] - hypo_depths[1:])  # (D-1, )
    hypo_xs = 1 / fx * hypo_depths  # (D, )
    hypo_ys = 1 / fy * hypo_depths  # (D, )

    grad_x = torch.abs(sdf_volume[:, :, :, :-1] - sdf_volume[:, :, :, 1:]) / hypo_xs[None, :, None, None]  # (B, D, H, W-1)
    grad_y = torch.abs(sdf_volume[:, :, :-1, :] - sdf_volume[:, :, 1:, :]) / hypo_ys[None, :, None, None]  # (B, D, H-1, W)
    grad_z = torch.abs(sdf_volume[:, :-1, :, :] - sdf_volume[:, 1:, :, :]) / z_diffs[None, :, None, None]  # (B, D-1, H, W)

    sdf_gradient = torch.stack([grad_x[:, :depth_num-1, :height-1, :width-1],
                                grad_y[:, :depth_num-1, :height-1, :width-1],
                                grad_z[:, :depth_num-1, :height-1, :width-1]], dim=1)  # (B, 3, D-1, H-1, W-1)

    sdf_gradient_norm = torch.norm(sdf_gradient, dim=1)  # (B, D-1, H-1, W-1)

    return sdf_gradient_norm


def sample_from_local_points(local_sample_points,sdf_gradient_norm):
    
    '''
    local_sample_points: [B,2*radius+1,H,W]
    sdf_gradient_norm: [B,D-1,H-1,W-1]
    '''
    local_sample_points =local_sample_points[:,:,:-1,:-1] # [B,2*radius+1,H,W]
    
    D = sdf_gradient_norm.shape[1]
    # valid mask
    sample_candidate_ceil = ste_ceil(local_sample_points)
    sample_candidate_floor = ste_floor(local_sample_points)
    sample_candidate_ceil = torch.clamp(sample_candidate_ceil,min=0,max=D-1)
    sample_candidate_floor = torch.clamp(sample_candidate_floor,min=0,max=D-1)
    # Linear interplotation
    floor_rate =(sample_candidate_ceil- local_sample_points)
    ceil_rate = 1.0 - floor_rate
    ceil_volume = torch.gather(sdf_gradient_norm,dim=1,index=sample_candidate_ceil.long())
    floor_volume = torch.gather(sdf_gradient_norm,dim=1,index=sample_candidate_floor.long())
    final_volume = ceil_volume*ceil_rate+ floor_volume*floor_rate
    
    return final_volume



def ste_ceil(x):
    return torch.ceil(x) - x.detach() + x

def ste_floor(x):
    return torch.floor(x) - x.detach() +x








if __name__=="__main__":
    sample = torch.rand(2,24,40,80)
    sdf_gradient_norm = SDF2Graident(sample)
    print(sdf_gradient_norm.shape)
    eki_loss = ((sdf_gradient_norm - torch.ones_like(sdf_gradient_norm))**2).mean()
    print(eki_loss)
