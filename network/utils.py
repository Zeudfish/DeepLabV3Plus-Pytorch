import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier,info_pro_param=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.info_pro_param = info_pro_param

        self.infopro_training = False
        
        if info_pro_param is not None:
            self.use_infopro = True
            self.loss_recons = info_pro_param.get('loss_recons', 0.5)
            self.loss_task = info_pro_param.get('loss_task', 1.0)
            
            # 中间监督头
            self.head_inter = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, 19 if 'num_classes' not in info_pro_param else info_pro_param['num_classes'], 
                         kernel_size=1) # 21 for VOC, 19 for Cityscapes
            )
            
            # 重建解码器
            self.decoder_inter = nn.Sequential(
                nn.Conv2d(1024, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(32, 12, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
            
            self.criterion_inter = nn.CrossEntropyLoss(ignore_index=255)
            self.bce_loss = nn.BCELoss()
        else:
            self.use_infopro = False
    
    def set_infopro_training(self, mode=True):
        """设置InfoPro训练模式"""
        self.infopro_training = mode
        
    def get_info_losses(self):
        """获取InfoPro损失"""
        return self.info_losses
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        if self.use_infopro and self.infopro_training and hasattr(self, '_labels'):
            # InfoPro训练模式
            labels = self._labels
            
            # 保存原始图像用于重建
            ini_image = x.clone()
            # 反归一化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            for i in range(3):
                ini_image[:, i, :, :] = ini_image[:, i, :, :] * std[i] + mean[i]
            
            # 检查backbone是否有forward_train1方法
            if hasattr(self.backbone, 'forward_train1'):
                # 第一部分前向传播
                x_inter = self.backbone.forward_train1(x)
                
                # 中间监督
                h, w = labels.shape[1:]
                scale_pred = F.interpolate(self.head_inter(x_inter), size=(h, w), 
                                         mode='bilinear', align_corners=False)
                loss_task = self.criterion_inter(scale_pred, labels) * self.loss_task
                
                # 重建损失
                recons = self.decoder_inter(x_inter)
                loss_recons = self.bce_loss(recons, ini_image) * self.loss_recons
                
                # 存储损失
                self.info_losses = {
                    'loss_task': loss_task,
                    'loss_recons': loss_recons,
                    'loss_inter': loss_task + loss_recons
                }
                
                # 断开梯度，继续第二部分
                x_inter_detached = x_inter.detach()
                features = self.backbone.forward_train2(x_inter_detached)
            else:
                # 如果backbone不支持分段训练，使用正常前向传播
                print("Warning: backbone doesn't support InfoPro training, using normal forward")
                features = self.backbone(x)
                self.info_losses = {}
            
            # 最终的分割输出
            x = self.classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x
        else:
            # 正常前向传播
            features = self.backbone(x)
            x = self.classifier(features)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            return x

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out
    def forward_train1(self, x):
        """InfoPro第一部分前向传播"""
        # 检查第一个模块是否有forward_train1方法
        first_module_name = list(self._modules.keys())[0]
        first_module = self._modules[first_module_name]
        
        if hasattr(first_module, 'forward_train1'):
            return first_module.forward_train1(x)
        else:
            raise NotImplementedError(f"Module {first_module_name} doesn't support forward_train1")
    
    def forward_train2(self, x):
        """InfoPro第二部分前向传播"""
        first_module_name = list(self._modules.keys())[0]
        first_module = self._modules[first_module_name]
        
        if hasattr(first_module, 'forward_train2'):
            features = first_module.forward_train2(x)
            # 转换为字典格式
            if not isinstance(features, dict):
                features = {'out': features}
            return features
        else:
            raise NotImplementedError(f"Module {first_module_name} doesn't support forward_train2")
