import torch
import torch.nn.functional as F
import numpy as np

def D1_metric(D_pred, D_gt):
    E = torch.abs(D_pred - D_gt)
    E_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(E_mask.float())




def P1_metric(D_pred, D_gt):
    E = torch.abs(D_pred - D_gt)
    E_mask = (E > 1)
    return torch.mean(E_mask.float())


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def Disparity_EPE_Loss(predicted_disparity,gt_disparity):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<192
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6

    epe_val = torch.abs(predicted_disparity*valid_mask-gt_disparity*valid_mask).sum()/(valid_mask.sum()+eps)
    return epe_val

def Disparity_EPE_Occ(predicted_disparity,gt_disparity,occ_mask):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<192
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6
    valid_mask = valid_mask * occ_mask

    epe_val = torch.abs(predicted_disparity*valid_mask-gt_disparity*valid_mask).sum()/(valid_mask.sum()+eps)
    return epe_val

def Disparity_EPE_Noc(predicted_disparity,gt_disparity,occ_mask,oof_mask):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<192
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6
    
    valid_mask = valid_mask.float()
    occ_mask = occ_mask.float()
    oof_mask = oof_mask.float()

    noc_mask = valid_mask - occ_mask - oof_mask
    noc_mask = torch.clamp(noc_mask,min=0,max=1)
    noc_mask = noc_mask.type_as(occ_mask)
    
    epe_val = torch.abs(predicted_disparity*noc_mask - gt_disparity*noc_mask).sum()/(valid_mask.sum()+eps)
    
    return epe_val



def D1_metric_Occ(predicted_disparity,gt_disparity,occ_mask,oof_mask):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<192
    valid_mask = valid_mask1 * valid_mask2
    valid_mask = valid_mask.float()
    
    
    eps =1e-6
    noc_mask = valid_mask - occ_mask - oof_mask
    noc_mask = torch.clamp(noc_mask,min=0,max=1)
    noc_mask = noc_mask.type_as(occ_mask)
    
    
    occ_mask = occ_mask *valid_mask
    occ_mask = occ_mask.float()
    
    oof_mask = oof_mask *valid_mask
    oof_mask = oof_mask.float()
    
    noc_mask = noc_mask *valid_mask
    noc_mask = noc_mask.float()
    
    occ_mask = torch.clamp(occ_mask,min=0,max=1)
    noc_mask = torch.clamp(noc_mask,min=0,max=1)
    oof_mask = torch.clamp(oof_mask,min=0,max=1)
    

    E = torch.abs(predicted_disparity*valid_mask- gt_disparity*valid_mask)
    E_mask = (E > 3) & (E / (gt_disparity*valid_mask).abs() > 0.05)
    D1_All = E_mask.sum()*1.0/(valid_mask.sum()+eps)
    
    E_Noc = torch.abs(predicted_disparity*noc_mask- gt_disparity*noc_mask)
    E_Noc_mask = (E_Noc > 3) & (E / (gt_disparity*noc_mask).abs() > 0.05)
    D1_Noc = E_Noc_mask.sum()*1.0/(noc_mask.sum()+eps)
    
    E_Ooc = torch.abs(predicted_disparity*occ_mask- gt_disparity*occ_mask)
    E_Ooc = torch.clamp(E_Ooc,min=0,max=100)
    E_Ooc_mask = (E_Ooc > 3) & (E / (gt_disparity*occ_mask).abs() > 0.05)
    D1_Ooc = E_Ooc_mask.sum()*1.0/(occ_mask.sum()+eps)
    
   
    
    E_Oof = torch.abs(predicted_disparity*oof_mask- gt_disparity*oof_mask)
    E_Oof = torch.clamp(E_Oof,min=0,max=100)
    E_Oof_mask = (E_Oof> 3)& (E / (gt_disparity*oof_mask).abs() > 0.05)
    D1_Oof = E_Oof_mask.sum()*1.0/(oof_mask.sum()+eps)
    
    
    return D1_All,D1_Ooc,D1_Oof,D1_Noc





def Disaprity_EPE_OOF(predicted_disparity,gt_disparity,oof_mask):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<192
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6
    valid_mask = valid_mask * oof_mask

    epe_val = torch.abs(predicted_disparity*valid_mask-gt_disparity*valid_mask).sum()/(valid_mask.sum()+eps)
    return epe_val



def Disparity_EPE_Loss_KITTI(predicted_disparity,gt_disparity):
    valid_mask1 = gt_disparity >0 
    valid_mask2 = gt_disparity<320
    valid_mask = valid_mask1 * valid_mask2
    eps =1e-6
    epe_val = torch.abs(predicted_disparity*valid_mask-gt_disparity*valid_mask).sum()/(valid_mask.sum()+eps)
    return epe_val




def P1_Value(predicted_disparity,gt_disparity):
    valid_mask = gt_disparity>0
    eps =1e-6
    E = torch.abs(predicted_disparity*valid_mask- gt_disparity*valid_mask)
    E_mask = (E>1)
    
    return E_mask.sum()*1.0/(valid_mask.sum()+eps)


def D1_metric(predicted_disparity,gt_disparity):
    valid_mask = gt_disparity>0
    eps =1e-6
    E = torch.abs(predicted_disparity*valid_mask- gt_disparity*valid_mask)
    E_mask = (E > 3) & (E / (gt_disparity*valid_mask).abs() > 0.05)
    
    return E_mask.sum()*1.0/(valid_mask.sum()+eps)




@torch.no_grad()
def Occlusion_EPE(predicted_occlusion,target_occlusion,disp_gt):
    mask = (disp_gt>0) & (disp_gt<192)
    predicted_occlusion = predicted_occlusion.float()[mask]
    target_occlusion = target_occlusion.float()[mask]
    return F.l1_loss(predicted_occlusion,target_occlusion,size_average=True)


@torch.no_grad()
def Occlusion_EPE_KITTI(predicted_occlusion,target_occlusion,disp_gt):
    mask = (disp_gt>0) & (disp_gt<320)
    predicted_occlusion = predicted_occlusion.float()[mask]
    target_occlusion = target_occlusion.float()[mask]
    return F.l1_loss(predicted_occlusion,target_occlusion,size_average=True)



@torch.no_grad()
def compute_iou(pred, occ_mask, target_disp):
    """
    compute IOU on occlusion
    :param pred: occlusion prediction [N,H,W]
    :param occ_mask: ground truth occlusion mask [N,H,W]
    :param loss_dict: dictionary of losses
    :param invalid_mask: invalid disparities (including occ and places without data), [N,H,W]
    """
    invalid_mask = (target_disp<0) & (target_disp>192)
    invalid_mask = invalid_mask + occ_mask
    invalid_mask = invalid_mask.bool()
    
    # threshold
    pred_mask = pred > 0.5
    # iou for occluded region
    inter_occ = torch.logical_and(pred_mask, occ_mask).sum()
    union_occ = torch.logical_or(torch.logical_and(pred_mask, ~invalid_mask), occ_mask).sum()
    # iou for non-occluded region
    inter_noc = torch.logical_and(~pred_mask, ~invalid_mask).sum()
    union_noc = torch.logical_or(torch.logical_and(~pred_mask, occ_mask), ~invalid_mask).sum()
    # aggregate
    iou = (inter_occ + inter_noc).float() / (union_occ + union_noc)
    return iou

