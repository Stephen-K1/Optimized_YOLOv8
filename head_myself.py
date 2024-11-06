# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from .block import DFL, Proto, ContrastiveHead, BNContrastiveHead
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"


# original head
class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


# class Detect(nn.Module):
#     """YOLOv8 Detect head for detection models."""

#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init

#     def __init__(self, nc=80, ch=()):
#         """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
#         )
#         self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#         '''下面两行为只修改8x输出的head, 看channel从64->128能不能提升小目标精度'''
#         self.cv2[0] = nn.Sequential(Conv(64, 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1))
#         self.cv3[0] = nn.Sequential(Conv(64, 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1))

#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return x

#         # Inference path
#         shape = x[0].shape  # BCHW
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape

#         if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
#             box = x_cat[:, : self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4 :]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

#         if self.export and self.format in ("tflite", "edgetpu"):
#             # Precompute normalization factor to increase numerical stability
#             # See https://github.com/ultralytics/ultralytics/issues/7371
#             grid_h = shape[2]
#             grid_w = shape[3]
#             grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
#             norm = self.strides / (self.stride[0] * grid_size)
#             dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
#         else:
#             dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)

#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

#     def decode_bboxes(self, bboxes, anchors):
#         """Decode bounding boxes."""
#         return dist2bbox(bboxes, anchors, xywh=True, dim=1)
    

class M_Detect(nn.Module):  # M表示modified，在这里修改自己想要的head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        # self.pixel_section = 3   # 多增加channel来定位某条边在像素点的哪里, 恢复原来直接注释掉这里就行
        self.epoch = 100
        # self.reg_max = 24
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        '''原来的head'''
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        # self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * 13, 1)) for x in ch)
        # self.no = nc + 4 * 13
        self.xywh = None   # NOTE 这个属性不要删掉，改成True或None
        # self.cv2 = nn.ModuleList()
        # self.cv2.append(nn.Sequential(Conv(ch[0], c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 40, 1)))
        # self.cv2.append(nn.Sequential(Conv(ch[1], c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 64, 1)))
        # self.cv2.append(nn.Sequential(Conv(ch[2], c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 80, 1)))
        # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.different_channel = None   # NOTE 这个属性不要删掉，改成True或None
        self.new_dfl = True    # 这个用来控制用原来的dfl还是dfl分支的输出由最大的两个one-hot加权决定, False为原来，不要删掉这个属性
        self.reg_loss = None   # 这个属性用来控制是否多用一个loss来衡量像素间的权重
        self.small_object = None
        self.iou_weight = True    # 是否针对小的iou的正样本增加权重
        if hasattr(self, 'pixel_section'):
            # c2 = 4 * self.reg_max + 4 * self.pixel_section   # 可注释掉，注释掉则只增加最后一个conv的输出channel，没注释掉则增加整个cv2的输出channel
            self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max + 4 * self.pixel_section, 1)) for x in ch)
            self.no = self.no + self.pixel_section * 4
        
        '''多加一个cv1分支预测anchor'''
        # self.cv1 = nn.ModuleList(nn.Sequential(Conv(x, 96, 3), Conv(96, 96, 3), nn.Conv2d(96, 4, 1)) for x in ch)
        # self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        # self.no = self.no + 4   # 原始no=80+64，在loss计算时会用到这个来reshape之类
        # self.new_dfl = True
        # 修改的结构如下
        '''下面两行为只修改8x输出的head, 看channel从64->128能不能提升小目标精度'''
        # self.cv2[0] = nn.Sequential(Conv(64, 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1))
        # self.cv3[0] = nn.Sequential(Conv(64, 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1))
        '''下面两行为只修改32x输出的head'''
        # self.cv2[2] = nn.Sequential(Conv(ch[2], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1))
        # self.cv3[2] = nn.Sequential(Conv(ch[2], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1))
        '''下面修改整个head, 看比只修改8x时小目标能提升多少'''
        # self.cv2[0] = nn.Sequential(Conv(ch[0], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1))
        # self.cv2[1] = nn.Sequential(Conv(ch[1], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1))
        # self.cv2[2] = nn.Sequential(Conv(ch[2], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1))
        # self.cv3[0] = nn.Sequential(Conv(ch[0], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1))
        # self.cv3[1] = nn.Sequential(Conv(ch[1], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1))
        # self.cv3[2] = nn.Sequential(Conv(ch[2], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1))
        '''8x输出辅助head'''
        # self.auxiliary_head_8x = nn.ModuleList()
        # self.auxiliary_head_8x.append(nn.Sequential(Conv(64, 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1)))
        # self.auxiliary_head_8x.append(nn.Sequential(Conv(64, 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1)))
        '''3个输出都加辅助head'''
        # self.auxiliary_head = nn.ModuleList()
        # self.auxiliary_head.append(nn.Sequential(Conv(ch[0], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1)))  # 8x
        # self.auxiliary_head.append(nn.Sequential(Conv(ch[0], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1)))
        # self.auxiliary_head.append(nn.Sequential(Conv(ch[1], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1)))  # 16x
        # self.auxiliary_head.append(nn.Sequential(Conv(ch[1], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1)))
        # self.auxiliary_head.append(nn.Sequential(Conv(ch[2], 256, 3), Conv(256, 64, 3), Conv(64, 64, 3), nn.Conv2d(c2, 64, 1)))  # 32x
        # self.auxiliary_head.append(nn.Sequential(Conv(ch[2], 256, 3), Conv(256, 256, 3), nn.Conv2d(256, self.nc, 1)))


    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if hasattr(self, 'auxiliary_head_8x'):
            feature_8x = x[0]
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            auxiliary_8x_output = torch.cat((self.auxiliary_head_8x[0](feature_8x), self.auxiliary_head_8x[1](feature_8x)), 1)   # (batch, 64, 80, 80) cat (batch, cls_num, 80, 80)
            x.append(auxiliary_8x_output)
        elif hasattr(self, 'auxiliary_head'):
            feature_8x, feature_16x, feature_32x = x
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            x.append(torch.cat((self.auxiliary_head[0](feature_8x), self.auxiliary_head[1](feature_8x)), 1))
            x.append(torch.cat((self.auxiliary_head[2](feature_16x), self.auxiliary_head[3](feature_16x)), 1))
            x.append(torch.cat((self.auxiliary_head[4](feature_32x), self.auxiliary_head[5](feature_32x)), 1))
        elif hasattr(self, 'cv1'):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv1[i](x[i]), self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        else:
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        try:
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        except:
            pass   # 上面有错即head的输出channel不同，为这个情况得到了相应的box和cls，下面也不需要用到x_cat
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            if hasattr(self, 'cv1'):
                offset, box, cls = x_cat.split((4, self.reg_max * 4, self.nc), 1)
            elif self.different_channel:
                c1, c2, c3 = x[0].shape[1] - self.nc, x[1].shape[1] - self.nc, x[2].shape[1] - self.nc
                cls = torch.cat([xi[:, -self.nc:].view(x[0].shape[0], self.nc, -1) for xi in x], 2)   # (batch, 80, 8400)
                box = [x[0][:, :c1].view(x[0].shape[0], c1, -1), x[1][:, :c2].view(x[0].shape[0], c2, -1), x[2][:, :c3].view(x[0].shape[0], c3, -1)]
            else:    
                box, cls = x_cat.split((self.no - self.nc, self.nc), 1)
        if hasattr(self, 'cv1'):
            dbox = self.self_decode(box.permute(0, 2, 1), offset.permute(0, 2, 1)).permute(0, 2, 1) * self.strides
        elif self.new_dfl:
            if hasattr(self, 'pixel_section'):
                dbox = self.self_decode_3(box.permute(0, 2, 1), self.pixel_section, self.epoch).permute(0, 2, 1) * self.strides
            elif isinstance(box, list):
                dbox = self.self_decode_2([b.permute(0, 2, 1) for b in box]).permute(0, 2, 1) * self.strides
            elif self.xywh:
                dbox = self.self_decode_2(box.permute(0, 2, 1), self.xywh).permute(0, 2, 1) * self.strides
            else:
                dbox = self.self_decode_2(box.permute(0, 2, 1)).permute(0, 2, 1) * self.strides
        else:
            dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        if self.small_object:
            classes = torch.tensor([29, 30, 39, 73, 9, 15]).cuda()
            y = y.permute(0, 2, 1)  # (batch, 8400, 84)
            pred_cls = y[..., 4:].max(dim=-1, keepdim=True)[1]  # (batch, 8400, 1)
            mask = torch.isin(pred_cls.squeeze(-1).int(), classes)
            temp = y[mask]
            temp[:, 2:4] /= 1.5
            y[mask] = temp
            y = y.permute(0, 2, 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    
    def self_decode(self, pred_dist, pred_offset):
        b, a, c = pred_dist.shape  # batch, anchors, channels
        if self.if_dfl:
            proj = torch.arange(16, dtype=torch.float, device=pred_dist.device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
        else:
            # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).max(dim=-1)[1].type(pred_dist.dtype)  # shape=(batch, 8400, 4), 得到坐标后dtype为int64, 转回原来的float16
            '''对(4, 8400, 4, 16), 下面取16这个维度相邻最大的两个相邻值的加权平均为输出, 即和dfl的loss计算对应, 得到(4, 8400, 4)'''
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).type(pred_dist.dtype)
            pred_dist = self.process_tensor(pred_dist)
        lt, rb = pred_dist.chunk(2, dim=-1)
        x1y1 = self.anchors.permute(1, 0) - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
        x2y2 = self.anchors.permute(1, 0) + rb

        pred_anchor = torch.cat((x1y1, x2y2), dim=-1)   # (batch, 8400, 4), xyxy, 大小为特征图大小
        return self.decode(pred_offset, pred_anchor)


    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)
        # TO_REMOVE = 1  # TODO remove
        widths = anchors[:, :, 2] - anchors[:, :, 0]   # (batch, 8400)
        heights = anchors[:, :, 3] - anchors[:, :, 1]  # (batch, 8400)
        ctr_x = (anchors[:, :, 2] + anchors[:, :, 0]) / 2   # (batch, 8400)
        ctr_y = (anchors[:, :, 3] + anchors[:, :, 1]) / 2   # (batch, 8400)
        wx, wy, ww, wh = (10., 10., 5., 5.)
        # wx, wy, ww, wh = (7.5, 7.5, 3.75, 3.75)
        dx = preds[:, :, 0] / wx   # (batch, 8400)
        dy = preds[:, :, 1] / wy
        dw = preds[:, :, 2] / ww
        dh = preds[:, :, 3] / wh
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_boxes = torch.zeros_like(preds)
        # pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w
        # pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h
        # pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w
        # pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h
        pred_boxes[:, :, 0] = pred_ctr_x
        pred_boxes[:, :, 1] = pred_ctr_y
        pred_boxes[:, :, 2] = pred_w
        pred_boxes[:, :, 3] = pred_h
        return pred_boxes   # 返回(batch, 8400, 4)，这里为xywh的形式，大小为特征图大小
    

    def process_tensor(self, pred_dist, xywh=None):
        if xywh:
            adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 4, 7)

            # Step 2: Find the indices of the maximum adjacent sum
            max_sum_indices = torch.argmax(adjacent_sums, dim=-1)  # Shape: (4, 8400, 4)

            # Step 3: Gather the two adjacent values corresponding to max_sum_indices
            indices1 = max_sum_indices   # (4, 8400, 4)
            indices2 = indices1 + 1
            gathered_values1 = torch.gather(pred_dist, dim=-1, index=indices1.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
            gathered_values2 = torch.gather(pred_dist, dim=-1, index=indices2.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
            summed_values = gathered_values1 + gathered_values2

            weights1 = gathered_values1 / summed_values  # Shape: (4, 8400, 4)
            weights2 = gathered_values2 / summed_values  # Shape: (4, 8400, 4)

            final_tensor = indices1 * weights1 + indices2 * weights2  # Shape: (4, 8400, 4)
        elif pred_dist.shape[-1] == 16:
            # Step 1: Get the sum of adjacent values in the 8 dimension
            adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 4, 7)

            # Step 2: Find the indices of the maximum adjacent sum
            max_sum_indices = torch.argmax(adjacent_sums, dim=-1)  # Shape: (4, 8400, 4)

            # Step 3: Gather the two adjacent values corresponding to max_sum_indices
            indices1 = max_sum_indices   # (4, 8400, 4)
            indices2 = indices1 + 1
            gathered_values1 = torch.gather(pred_dist, dim=-1, index=indices1.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
            gathered_values2 = torch.gather(pred_dist, dim=-1, index=indices2.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
            summed_values = gathered_values1 + gathered_values2

            weights1 = gathered_values1 / summed_values  # Shape: (4, 8400, 4)
            weights2 = gathered_values2 / summed_values  # Shape: (4, 8400, 4)

            final_tensor = indices1 * weights1 + indices2 * weights2  # Shape: (4, 8400, 4)
        
        else:  # NOTE 这里暂时试一下10个像素类别，原来最多预测到15，比例为2:3
            adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 4, 10)

            # Step 2: Find the indices of the maximum adjacent sum
            max_sum_indices = torch.argmax(adjacent_sums, dim=-1)  # Shape: (4, 8400, 4)

            # Step 3: Gather the two adjacent values corresponding to max_sum_indices
            indices1 = max_sum_indices   # (4, 8400, 4)
            indices2 = indices1 + 1
            gathered_values1 = torch.gather(pred_dist, dim=-1, index=indices1.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
            gathered_values2 = torch.gather(pred_dist, dim=-1, index=indices2.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
            summed_values = gathered_values1 + gathered_values2

            weights1 = gathered_values1 / summed_values  # Shape: (4, 8400, 4)
            weights2 = gathered_values2 / summed_values  # Shape: (4, 8400, 4)

            final_tensor = 15 / (pred_dist.shape[-1] - 1) * (indices1 * weights1 + indices2 * weights2)  # Shape: (4, 8400, 4)

        return final_tensor
    
    '''这个是原来的head, 但是dfl输出只由最大的两个相邻的one-hot的加权决定时使用, 即if not hasattri(self, 'cv1') and self.new_dfl=True时使用'''
    def self_decode_2(self, pred_dist, xywh=None):
        if xywh:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            xy = pred_dist[..., :2]   # (batch, 8400, 2)
            wh = pred_dist[..., 2:]   # (batch, 8400, 62)
            pred_xy = xy.sigmoid() * 2 - 1 + self.anchors.permute(1, 0)  # (batch, 8400, 2)
            pred_wh = wh.view(b, a, 2, wh.shape[-1] // 2).softmax(3).type(pred_dist.dtype)  # softmax values, (batch, 8400, 2, 31)
            pred_wh = self.process_tensor(pred_wh, xywh)   # (batch, 8400, 2)
            pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)  # (batch, 8400, 4)
            pred_xyxy = xywh2xyxy(pred_xywh)
            x1y1 = pred_xyxy[..., :2]
            x2y2 = pred_xyxy[..., 2:]
        elif isinstance(pred_dist, list):
            feat1 = pred_dist[0].view(pred_dist[0].shape[0], pred_dist[0].shape[1], 4, pred_dist[0].shape[2] // 4).softmax(3).type(pred_dist[0].dtype)  # (b, 6400, 4, 13)
            feat2 = pred_dist[1].view(pred_dist[1].shape[0], pred_dist[1].shape[1], 4, pred_dist[1].shape[2] // 4).softmax(3).type(pred_dist[0].dtype)  # (b, 1600, 4, 16)
            feat3 = pred_dist[2].view(pred_dist[2].shape[0], pred_dist[2].shape[1], 4, pred_dist[2].shape[2] // 4).softmax(3).type(pred_dist[0].dtype)  # (b, 400, 4, 31)
            out1 = self.process_tensor(feat1)
            out2 = self.process_tensor(feat2)
            out3 = self.process_tensor(feat3)
            pred_dist = torch.cat([out1, out2, out3], dim=1)
            lt, rb = pred_dist.chunk(2, dim=-1)
            x1y1 = self.anchors.permute(1, 0) - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
            x2y2 = self.anchors.permute(1, 0) + rb
        else:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).max(dim=-1)[1].type(pred_dist.dtype)  # shape=(batch, 8400, 4), 得到坐标后dtype为int64, 转回原来的float16
            '''对(4, 8400, 4, 16), 下面取16这个维度相邻最大的两个相邻值的加权平均为输出, 即和dfl的loss计算对应, 得到(4, 8400, 4)'''
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).type(pred_dist.dtype)
            # pred_dist = torch.sigmoid(pred_dist.view(b, a, 4, c // 4)).type(pred_dist.dtype)  # sigmoid values, (batch, 8400, 4, 16)
            pred_dist = self.process_tensor(pred_dist)
            lt, rb = pred_dist.chunk(2, dim=-1)
            x1y1 = self.anchors.permute(1, 0) - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
            x2y2 = self.anchors.permute(1, 0) + rb

        # 默认取xywh的形式，validate时head的forward才会走到这里，train时得到最后的conv输出就传给loss.py了
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=-1)  # xywh bbox
    

    def self_decode_3(self, pred_dist, pixel_section, epoch=100):    
        dfl_prediction = pred_dist[..., :64]
        section_prediction = pred_dist[..., 64:]
        d = section_prediction.shape[-1]
        b, a, c = dfl_prediction.shape  # batch, anchors, channels
        # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).max(dim=-1)[1].type(pred_dist.dtype)  # shape=(batch, 8400, 4), 得到坐标后dtype为int64, 转回原来的float16
        '''对(4, 8400, 4, 16), 下面取16这个维度相邻最大的两个相邻值的加权平均为输出, 即和dfl的loss计算对应, 得到(4, 8400, 4)'''
        dfl_prediction = dfl_prediction.view(b, a, 4, c // 4).softmax(3).type(pred_dist.dtype)  # softmax values, (batch, 8400, 4, 16)
        section_prediction = section_prediction.view(b, a, 4, d // 4).softmax(3).type(pred_dist.dtype)  # softmax values, (batch, 8400, 4, 10)
        # dfl_prediction = torch.sigmoid(dfl_prediction.view(b, a, 4, c // 4)).type(pred_dist.dtype)  # sigmoid values, (batch, 8400, 4, 16)
        # section_prediction = torch.sigmoid(section_prediction.view(b, a, 4, d // 4)).type(pred_dist.dtype)  # sigmoid values, (batch, 8400, 4, 16)
        pred_dist = self.process_tensor_2(dfl_prediction, section_prediction, pixel_section, epoch)  # (batch, 8400, 4)
        lt, rb = pred_dist.chunk(2, dim=-1)
        x1y1 = self.anchors.permute(1, 0) - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
        x2y2 = self.anchors.permute(1, 0) + rb

        # 默认取xywh的形式，validate时head的forward才会走到这里，train时得到最后的conv输出就传给loss.py了
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=-1)  # xywh bbox
    

    def process_tensor_2(self, pred_dist, section_prediction, pixel_section, epoch):
        # Step 1: Get the sum of adjacent values in the 8 dimension
        adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 4, 15)
        adjacent_sums_2 = section_prediction[..., :-1] + section_prediction[..., 1:]  # Shape: (4, 8400, 4, 10)

        # Step 2: Find the indices of the maximum adjacent sum
        pixel = torch.argmax(adjacent_sums, dim=-1)  # Shape: (4, 8400, 4)
        max_sum_indices = torch.argmax(adjacent_sums_2, dim=-1)  # Shape: (4, 8400, 4)

        indices1 = max_sum_indices   # (4, 8400, 4)
        indices2 = indices1 + 1
        gathered_values1 = torch.gather(section_prediction, dim=-1, index=indices1.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
        gathered_values2 = torch.gather(section_prediction, dim=-1, index=indices2.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
        summed_values = gathered_values1 + gathered_values2

        weights1 = gathered_values1 / summed_values  # Shape: (4, 8400, 4)
        weights2 = gathered_values2 / summed_values  # Shape: (4, 8400, 4)

        section = indices1 * weights1 + indices2 * weights2  # Shape: (4, 8400, 4)

        # '''这行为当只用一个one-hot决定某个pixel时, 用两个one-hot决定则注释掉'''
        # pixel = torch.argmax(pred_dist[..., :-1], dim=-1)  # Shape: (4, 8400, 4)

        output = pixel +  section * 1 / (pixel_section - 1)


        '''原始new_dfl和新的区间同时决定output, 各占1/2'''
        indices1 = pixel   # (4, 8400, 4)
        indices2 = indices1 + 1
        gathered_values1 = torch.gather(pred_dist, dim=-1, index=indices1.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
        gathered_values2 = torch.gather(pred_dist, dim=-1, index=indices2.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 4)
        summed_values = gathered_values1 + gathered_values2

        weights1 = gathered_values1 / summed_values  # Shape: (4, 8400, 4)
        weights2 = gathered_values2 / summed_values  # Shape: (4, 8400, 4)

        output_2 = indices1 * weights1 + indices2 * weights2  # Shape: (4, 8400, 4)

        # output = 0.5 * (output + output_2)
        output = 0.9 * output_2 + 0.1 * output

        # weight = 0.9 - 0.4 * epoch / 100   # 100个epoch, 0.9 -> 0.5

        # output = weight * output_2 + (1 - weight) * output

        return output


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)
