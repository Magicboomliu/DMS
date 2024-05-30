import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.disparity_warper import disp_warp

# conv3x3 + BN + relu
def conv(in_planes, out_planes, kernel_size=3, stride=1, batchNorm=False):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
# conv3x3 + BN
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
        nn.BatchNorm2d(out_planes)
    )

# simple conv3x3 only    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, stride=stride, kernel_size=3, padding=1, bias=False)

# deconv : upsample to double
def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )
# conv + relu
def conv_Relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.ReLU(True)
    )
# conv3d + BatchNorm
def convbn_3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))
    
    
# Simple Residual Prediction: Left-Right Images, Disparity Warped Error
class res_submodule(nn.Module):
    def __init__(self, scale, input_layer, out_planes=64):
        super(res_submodule, self).__init__()
        #self.resample = Resample2d()
        
        # Avgerage Pooling
        self.pool = nn.AvgPool2d(2**scale, 2**scale)
        # Left + Right + Warped Error + Coarse Disparity : 10
        # self.conv1: Downsample to input's 1/2, channels doubled
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        # Self.conv2: aggregation: Size not change
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2),
            nn.ReLU(True)
        )
        # Self.conv3: Downsample to input's 1/4, channels doubled again
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )
        # Self.conv4: aggregation: Size not change
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_planes*4, out_planes*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes*4),
            nn.ReLU(True)
        )

        # Self.conv5: Upsample to former's 1/2, channels half
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*4, out_planes*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes*2)
        )

        # Self.conv6; Upsample to input's Size, channels half
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(out_planes*2, out_planes, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        # Skip connection1
        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layer+10, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Skip connection2 
        self.redir2 = nn.Sequential(
            nn.Conv2d(out_planes*2, out_planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Residual prediction
        self.res = nn.Conv2d(out_planes, 1, 1, 1, bias=False)
    
    def forward(self, left, right, disp, feature):

        # Current Scale for interpolation
        scale = left.size()[2] / disp.size()[2]
        # Left right Image Pooling
        left = self.pool(left)
        right = self.pool(right)

        # Align disparity to the current scale for image warped error computaion

        disp_ = disp / scale            # align the disparity to the proper scale
        # flow = torch.cat((disp_, dummy_flow), dim=1)
        left_rec,mask = disp_warp(right,disp_)
        # Calculate the disparity error
        error_map = left_rec -left
         
        # Downsample to 1/2
        conv1 = self.conv1(torch.cat((left, right, disp_, error_map, feature), dim=1))
        # Aggregation
        conv2 = self.conv2(conv1)
        # Downsample to 1/4
        conv3 = self.conv3(conv2)
        # Aggregation
        conv4 = self.conv4(conv3)
        # Upsample+Skip Connection to 1/2
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # Upsample + Skip Conncetion to 1/2
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((left, right, disp_, error_map, feature), dim=1)))
        # Recover to the Original Scale
        res = self.res(conv6) * scale
        return res





class res_disparity_refinement(nn.Module):
    def __init__(self,input_layers,cost_volume_disp,hidden_dim):
        super().__init__()
        self.cost_volume_projection = nn.Sequential(
            nn.Conv2d(cost_volume_disp,hidden_dim,3,1,1,bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)
        )
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_layers+hidden_dim+1, hidden_dim*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(True)
        )

        # Self.conv5: Upsample to former's 1/2, channels half
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim*2)
        )

        # Self.conv6; Upsample to input's Size, channels half
        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )

        # Skip connection1
        self.redir1 = nn.Sequential(
            nn.Conv2d(input_layers+hidden_dim+1, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Skip connection2 
        self.redir2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Residual prediction
        self.res = nn.Conv2d(hidden_dim, 1, 1, 1, bias=False)

    
    def forward(self,local_cost_volume,cur_disp,feat):
        
        cost_volume_width = local_cost_volume.shape[-1]
        cur_disp_width = cur_disp.shape[-1]
        
        scale = cost_volume_width//cur_disp_width
        if scale!=1:
            cur_disp = F.interpolate(cur_disp,scale_factor=scale,mode='bilinear',align_corners=False) * scale
        
        aggregated_cost_volume = self.cost_volume_projection(local_cost_volume)
        
        # Downsample to 1/2
        conv1 = self.conv1(torch.cat((aggregated_cost_volume, feat, cur_disp), dim=1))
        # Aggregation
        conv2 = self.conv2(conv1)
        # Downsample to 1/4
        conv3 = self.conv3(conv2)
        # Aggregation
        conv4 = self.conv4(conv3)
        # Upsample+Skip Connection to 1/2
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # Upsample + Skip Conncetion to 1/2
        conv6 = F.relu(self.conv6(conv5) + self.redir1(torch.cat((aggregated_cost_volume, feat, cur_disp), dim=1)))
        # Recover to the Original Scale
        res = self.res(conv6)
        
        cur_disp = F.relu(cur_disp+res) 
        
        
        return cur_disp




if __name__=="__main__":
    pass