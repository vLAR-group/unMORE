import copy
import random
import math
import torchvision
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms as T


class SoftPositionEmbed(nn.Module):
    def __init__(self, device, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution).to(device)

    def forward(self, inputs):
        grid = self.embedding(self.grid).permute(0,3,1,2)
        return inputs + grid

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sin(x)

class ObjectnessNet(nn.Module):
    def __init__(
        self,
        device,
        image_size,
        backbone_type,
        args=None
    ):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.backbone_type = backbone_type
        self.args = args

        if backbone_type == 'resnet50':
            from torchvision.models import resnet50
            if self.args.use_seperate_backbone:
                sdf_fcn_backbone = torchvision.models.resnet50(pretrained=self.args.pretrain_weights, replace_stride_with_dilation=self.args.replace_stride_with_dilation)
                self.sdf_backbone = _fcn_resnet(backbone=sdf_fcn_backbone, num_classes=self.args.num_classes, aux=None)
                centerness_fcn_backbone = torchvision.models.resnet50(pretrained=self.args.pretrain_weights, replace_stride_with_dilation=self.args.replace_stride_with_dilation)
                self.centerness_backbone = _fcn_resnet(backbone=centerness_fcn_backbone, num_classes=self.args.num_classes, aux=None)
            else:
                fcn_backbone = torchvision.models.resnet50(pretrained=self.args.pretrain_weights, replace_stride_with_dilation=self.args.replace_stride_with_dilation)
                self.backbone = _fcn_resnet(backbone=fcn_backbone, num_classes=self.args.num_classes, aux=None)
            feat_dim = self.args.num_classes
        elif backbone_type == 'dpt_large':
            from models.dpt.models import DPT
            self.backbone = DPT(
                    head=None,
                    features=256,
                    backbone="vitl16_384",
                    readout="project",
                    channels_last=False,
                    use_bn=False,
                    enable_attention_hooks=False,
                )
            feat_dim = 256
        elif backbone_type == 'dpt_hybrid':
            from models.dpt.models import DPT
            if self.args.use_seperate_backbone:
                self.sdf_backbone = DPT(
                    head=None,
                    features=256,
                    backbone="vitb_rn50_384",
                    readout="project",
                    channels_last=False,
                    use_bn=False,
                    enable_attention_hooks=False,
                )
                self.centerness_backbone = DPT(
                    head=None,
                    features=256,
                    backbone="vitb_rn50_384",
                    readout="project",
                    channels_last=False,
                    use_bn=False,
                    enable_attention_hooks=False,
                )
            else:
                self.backbone = DPT(
                    head=None,
                    features=256,
                    backbone="vitb_rn50_384",
                    readout="project",
                    channels_last=False,
                    use_bn=False,
                    enable_attention_hooks=False,
                )
            feat_dim = 256
        else:
            raise NotImplementedError
        ## prediction head for Object Center Field
        self.center_field_prediction_head = nn.Sequential(
                        torch.nn.Conv2d(in_channels=feat_dim, out_channels=512, kernel_size=1, padding=0),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=1, padding=0),
                        )
        ## prediction head for Object Boundary Distance Field
        if self.args.use_bg_sdf:
            if self.args.sdf_activation == 'sine':
                self.sdf_prediction_head = nn.Sequential(
                        torch.nn.Conv2d(in_channels=feat_dim, out_channels=512, kernel_size=1, padding=0),
                        torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                        torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0),
                        torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0),
                        SinActivation(),
                        )
            elif self.args.sdf_activation == 'tanh':
                self.sdf_prediction_head = nn.Sequential(
                        torch.nn.Conv2d(in_channels=feat_dim, out_channels=512, kernel_size=1, padding=0),
                        torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                        torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0),
                        torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0),
                        torch.nn.Tanh(),
                        )
            elif self.args.sdf_activation == None:
                self.sdf_prediction_head = nn.Sequential(
                        torch.nn.Conv2d(in_channels=feat_dim, out_channels=512, kernel_size=1, padding=0),
                        torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                        torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0),
                        torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0),
                        )
            elif self.args.sdf_activation == 'relu':
                self.sdf_prediction_head = nn.Sequential(
                        torch.nn.Conv2d(in_channels=feat_dim, out_channels=512, kernel_size=1, padding=0),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0),
                        )
            else:
                raise NotImplementedError
        else:
            self.sdf_prediction_head = nn.Sequential(
                        torch.nn.Conv2d(in_channels=feat_dim, out_channels=512, kernel_size=1, padding=0),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0),
                        nn.ReLU(),
                        torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, padding=0),
                        )
        

    def forward(self, images):
        if 'resnet' in self.backbone_type or 'res2net' in self.backbone_type or 'resnext' in self.backbone_type:
            images_feat_dict = self.backbone(images)
            images_feat = images_feat_dict['out'] ## [B, C, H, W]
        elif self.backbone_type == 'dpt_large' or self.backbone_type == 'dpt_hybrid':
            images_feat = self.backbone(images)
        else:
            raise NotImplementedError
        out_dict = {}
        ## predict Object Center Field
        center_fields = self.center_field_prediction_head(images_feat) ## [B, 2, H, W]
        out_dict['center_fields'] = center_fields
        ## predict Object Boundary Distance Field
        sdf_maps = self.sdf_prediction_head(images_feat) ## [B, 1, H, W]
        out_dict['sdf_maps'] = sdf_maps

        return out_dict

    '''
    Help Function
    '''
    def get_prediction(self, images):
        if 'resnet' in self.backbone_type or 'res2net' in self.backbone_type or 'resnext' in self.backbone_type:
            images_feat_dict = self.backbone(images)
            images_feat = images_feat_dict['out'] ## [B, C, H, W]
        elif self.backbone_type == 'dpt_large' or self.backbone_type == 'dpt_hybrid':
            images_feat = self.backbone(images)
        else:
            raise NotImplementedError
        out_dict = {}
        ## predict Object Center Field
        center_fields = self.center_field_prediction_head(images_feat) ## [B, 2, H, W]
        out_dict['center_fields'] = center_fields
        ## predict Object Boundary Distance Field
        sdf_maps = self.sdf_prediction_head(images_feat) ## [B, 1, H, W]
        out_dict['sdf_maps'] = sdf_maps
        return out_dict

class Binary_Classifier(nn.Module):
    def __init__(
        self,
        device,
        image_size,
        args=None
    ):
        super().__init__()
        self.image_size = image_size
        self.device = device
        self.args = args
        self.classifier_backbone = torchvision.models.resnet50(pretrained=False)
        self.binary_classification_head = torch.nn.Linear(1000, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        prediction = self.binary_classification_head(self.classifier_backbone(images))
        prediction = self.sigmoid(prediction)
        return prediction ## [B, 2]


if __name__ == '__main__':
    vote = VOTE(image_size=448, device='cuda:0')