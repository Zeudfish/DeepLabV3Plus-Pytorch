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
    返回骨干网络的中间特征，并在 InfoPro 训练模式下
    调用 backbone 的 forward_train1 / forward_train2。
    """
    def __init__(self, model, return_layers, hrnet_flag: bool = False):
        # ---------- 检查 ----------
        if not set(return_layers).issubset(name for name, _ in model.named_children()):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        # ---------- 收集要保存的子层 ----------
        layers = OrderedDict()
        remaining = dict(return_layers)          # copy
        for name, module in model.named_children():
            layers[name] = module
            remaining.pop(name, None)
            if not remaining:
                break

        # 先调用父类构造函数，完成 Module 初始化
        super().__init__(layers)

        # 现在再保存对原始 backbone 的引用
        self.orig_model = model
        self.return_layers = return_layers       # {'layer4': 'out', ...}

    # -------------------- 普通推理 --------------------
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():  # self.named_children() 也行
            if self.hrnet_flag and name.startswith('transition'):
                if name == 'transition1':
                    x = [m(x) for m in module]
                else:
                    x.append(module(x[-1]))
            else:
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if self.hrnet_flag and name == 'stage4':          # HRNet cat
                    h, w = x[0].shape[2:]
                    ups = [F.interpolate(t, size=(h, w),
                                          mode='bilinear',
                                          align_corners=False)
                           for t in x[1:]]
                    x = torch.cat([x[0], *ups], dim=1)
                out[out_name] = x
        return out

    # -------------------- InfoPro 前半段 --------------------
    def forward_train1(self, x):
        if hasattr(self.orig_model, 'forward_train1'):
            return self.orig_model.forward_train1(x)
        raise NotImplementedError("Backbone does not implement forward_train1")

    # -------------------- InfoPro 后半段 --------------------
    def forward_train2(self, x):
        if hasattr(self.orig_model, 'forward_train2'):
            feats = self.orig_model.forward_train2(x)
            return feats if isinstance(feats, dict) else {'out': feats}
        raise NotImplementedError("Backbone does not implement forward_train2")

 
