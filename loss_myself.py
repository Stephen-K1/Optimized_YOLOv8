# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.sigmoid = True   # 指定当有pixel section预测时，用交叉熵还是二值交叉熵

    # pred_dist.shape=(batch, 8400, 64), raw output of model
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, new_dfl=False, 
                pixel_section=None, xywh=None, reg_loss=None, iou_weight=None, epoch=0):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # target_scores.shape=(4, 8400, 80), weight=(576, 1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        if iou_weight:
            mask = (iou >= 0.2) & (iou <= 0.7)
            weight_2 = torch.ones_like(iou, device=iou.device)
            # weight_2[mask] += torch.exp(iou[mask] * -5)
            weight_2[mask] += 1 - iou[mask]
            loss_iou = ((1.0 - iou) * weight * weight_2).sum() / target_scores_sum
        else:
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if new_dfl:  # 直接用交叉熵计算损失，默认使用dfl损失
            if pixel_section != None:  # 是否有pixel section预测
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                pred_section = pred_dist[..., 64:]
                pred_dist = pred_dist[..., :64]
                loss_dfl = self._new_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), pred_section[fg_mask].view(-1, pixel_section), target_ltrb[fg_mask], pixel_section, epoch) * weight
                # loss_dfl = self._bce_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight  # weight.shape=(num_positive, 1)
                loss_dfl = loss_dfl.sum() / target_scores_sum
            elif xywh:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                anchor_points_repeat = anchor_points.unsqueeze(dim=0).repeat_interleave(fg_mask.shape[0], dim=0)
                loss_dfl = self._xywh_loss(pred_dist[fg_mask], target_ltrb[fg_mask], anchor_points_repeat[fg_mask], target_bboxes[fg_mask]) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            # 不管new_dfl是否True, 直接用原dfl来计算损失
            elif isinstance(pred_dist, list):    # NOTE list表示每个head的box的输出channel不一样
                s1, s2, s3 = (pred_dist[0].shape[2] - 4) / 60, (pred_dist[1].shape[2] - 4) / 60, (pred_dist[2].shape[2] - 4) / 60
                weight_2 = [target_scores.sum(-1)[:, :pred_dist[0].shape[1]][fg_mask[:, :pred_dist[0].shape[1]]], target_scores.sum(-1)[:, pred_dist[0].shape[1]:(pred_dist[1].shape[1]+pred_dist[0].shape[1])][fg_mask[:, pred_dist[0].shape[1]:(pred_dist[1].shape[1]+pred_dist[0].shape[1])]],target_scores.sum(-1)[:, (pred_dist[1].shape[1]+pred_dist[0].shape[1]):][fg_mask[:, (pred_dist[1].shape[1]+pred_dist[0].shape[1]):]]]
                weight_2 = torch.cat(weight_2, dim=0).unsqueeze(dim=-1)
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)  # (batch, 8400, 4)
                target_ltrb[:, :pred_dist[0].shape[1]] *= s1
                target_ltrb[:, pred_dist[0].shape[1]:(pred_dist[1].shape[1]+pred_dist[0].shape[1])] *= s2
                target_ltrb[:, (pred_dist[1].shape[1]+pred_dist[0].shape[1]):] *= s3
                pred = [pred_dist[0][fg_mask[:, :pred_dist[0].shape[1]]], pred_dist[1][fg_mask[:, pred_dist[0].shape[1]:(pred_dist[1].shape[1]+pred_dist[0].shape[1])]], pred_dist[2][fg_mask[:, (pred_dist[1].shape[1]+pred_dist[0].shape[1]):]]]
                target =[target_ltrb[:, :pred_dist[0].shape[1]][fg_mask[:, :pred_dist[0].shape[1]]], target_ltrb[:, pred_dist[0].shape[1]:(pred_dist[1].shape[1]+pred_dist[0].shape[1])][fg_mask[:, pred_dist[0].shape[1]:(pred_dist[1].shape[1]+pred_dist[0].shape[1])]], target_ltrb[:, (pred_dist[1].shape[1]+pred_dist[0].shape[1]):][fg_mask[:, (pred_dist[1].shape[1]+pred_dist[0].shape[1]):]]]
                loss_dfl = self._df_loss_2(pred, target) * weight_2  # weight.shape=(num_positive, 1)
                loss_dfl = loss_dfl.sum() / target_scores_sum
            elif reg_loss:  
                '''NOTE 像素点间多了一个loss'''
                scale = (pred_dist.shape[-1] - 4) / 60
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max) * scale
                anchor_points_repeat = anchor_points.unsqueeze(dim=0).repeat_interleave(fg_mask.shape[0], dim=0)
                loss_dfl= self._reg_loss(pred_dist[fg_mask].view(-1, pred_dist.shape[-1] // 4), target_ltrb[fg_mask], anchor_points_repeat[fg_mask]) * weight  # weight.shape=(num_positive, 1)
                loss_dfl = loss_dfl.sum() / target_scores_sum
            elif self.reg_max == 15:   # 这里表示最原始的loss
                '''NOTE original'''
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                if iou_weight:
                    loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight * weight_2  # weight.shape=(num_positive, 1)
                else:
                    loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight  # weight.shape=(num_positive, 1)
                loss_dfl = loss_dfl.sum() / target_scores_sum
                '''NOTE modified'''
                # target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                # loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask] * 0.75) * weight  # weight.shape=(num_positive, 1)
                # loss_dfl = self._bce_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight  # weight.shape=(num_positive, 1)
            else:  # NOTE deprecated, 这里暂时实现dfl=31的，缩放回16
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max * 1.25)
                # loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask] / 1.25) * weight  # weight.shape=(num_positive, 1)
                loss_dfl = self._bce_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask] / 1.25) * weight  # weight.shape=(num_positive, 1)
                # target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max * 0.5)  # 这里传进的reg_max用于将gt_box限定在哪个范围
                # loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask] * 2) * weight  # weight.shape=(num_positive, 1)
                loss_dfl = loss_dfl.sum() / target_scores_sum
        
        else:
            # DFL loss
            if self.use_dfl:
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


    @staticmethod
    def _df_loss(pred_dist, target):  # pred_dist.shape=(576*4, 16), target.shape=(576, 4)
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (    # F.cross_entropy(pred_dist, tl.view(-1), reduction="none").shape=(pred_dist.shape[0])
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl   # shape=target.shape=(576, 4)
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)  # shape=(576, 1)
    

    @staticmethod
    def _reg_loss(pred_dist, target, anchor_points):  # pred_dist.shape=(576*4, 16), target.shape=(576, 4)
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left, (576, 4)
        tr = tl + 1  # target right, (576, 4)
        wl = tr - target  # weight left, (576, 4)
        wr = 1 - wl  # weight right, (576, 4)
        dfl_loss = (    # F.cross_entropy(pred_dist, tl.view(-1), reduction="none").shape=(pred_dist.shape[0])
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl   # shape=target.shape=(576, 4)
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)   # shape=(576, 1)

        '''focal loss'''
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
        focal_loss_l = 0.25 * (1 - torch.exp(-loss_l)) ** 2 * loss_l
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        focal_loss_r = 0.25 * (1 - torch.exp(-loss_r)) ** 2 * loss_r
        focal_loss = (focal_loss_l + focal_loss_r).mean(dim=-1, keepdim=True)

        pred_dist = pred_dist.softmax(dim=-1)  # (576*4, 16)
        tl = tl.view(-1).unsqueeze(-1)  # (576*4, 1)
        tr = tr.view(-1).unsqueeze(-1)
        pred_l = torch.gather(pred_dist, 1, tl).squeeze(1)  # (576*4)
        pred_r = torch.gather(pred_dist, 1, tr).squeeze(1)  # (576*4)
        # (576*4) -> (576, 4)
        # sum_loss = (1 - (pred_l + pred_r)).view(target.shape).mean(dim=-1, keepdim=True)  # 这两个相加应该=1，这里用类似iou损失计算，这个损失也有利于提升正确分类的值
        # t = torch.ones_like(pred_l)   # 直接让target=1，然后后面乘相应scale，避免抑制正确分类的值
        # reg_loss = (
        #     F.l1_loss(pred_l, t, reduction='none').view(target.shape) * wl
        #     + F.l1_loss(pred_r, t, reduction='none').view(target.shape)).mean(dim=-1, keepdim=True) * wr

        # reg_loss = (
        #     F.mse_loss(pred_l, wl.view(-1), reduction='none').view(target.shape)
        #     + F.mse_loss(pred_r, wr.view(-1), reduction='none').view(target.shape)).mean(dim=-1, keepdim=True)

        '''Wasserstein Distance Loss'''
        # pred_ltrb = process_tensor(pred_dist).view(target.shape)
        # wdl = F.l1_loss(pred_ltrb, target, reduction='none').mean(dim=-1, keepdim=True)

        '''mse loss for zero classes'''
        # false_classes = torch.ones_like(pred_dist)  # (576*4, 16)
        # batch_index = torch.arange(pred_dist.shape[0]).unsqueeze(1)
        # false_classes[batch_index, tl] = 0
        # false_classes[batch_index, tr] = 0
        # false_pred = pred_dist[false_classes.bool()].view(-1, pred_dist.shape[-1]-2)   # (576*4, 14)
        # false_target = torch.zeros_like(false_pred)
        # false_loss = F.mse_loss(false_pred, false_target, reduction='none').mean(-1).view(target.shape).mean(dim=-1, keepdim=True)

        return dfl_loss + focal_loss * 0.5
        # return (dfl_loss + sum_loss) * 0.75 # + sum_loss
    

    '''为xywh格式设置的loss, xy根据anchor-based得到, wh沿用dfl的形式'''
    # anchor_points.shape=(576, 2), target_bboxes.shape=(576, 4), xyxy, feature map size, max=80
    def _xywh_loss(self, pred_dist, target, anchor_points, target_bboxes):  # pred_dist.shape=(576, 64), target.shape=(576, 4)
        pred_wh = pred_dist[..., 2:].contiguous().view(-1, 31)  # (576, 62) -> (576*2, 31)
        target_wh = target[:, [2, 1]] + target[:, [0, 3]]  # (576, 2)
        tl = target_wh.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target_wh  # weight left
        wr = 1 - wl  # weight right
        loss_wh = (F.cross_entropy(pred_wh, tl.view(-1), reduction="none").view(tl.shape) * wl + 
                   F.cross_entropy(pred_wh, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)   # (576, 1)
        pred_xy = pred_dist[..., :2]  # (576, 2)
        x = (target_bboxes[:, 0] + target_bboxes[:, 2]) / 2  # x center
        y = (target_bboxes[..., 1] + target_bboxes[..., 3]) / 2  # y center
        target_xy = (torch.stack([x, y], dim=-1) - anchor_points + 1) / 2  # (576, 2)
        # bce loss
        # loss_xy = self.bce(pred_xy, target_xy.to(target.dtype)).mean(-1, keepdim=True)  # (576, 1)
        # l2 loss
        loss_xy = F.mse_loss(pred_xy, target_xy, reduction='none').mean(dim=-1, keepdim=True)

        return loss_xy + loss_wh


    '''这个当不同下采样的head有不同大小的box的大小时使用'''
    @staticmethod
    def _df_loss_2(pred_dist, target):  # pred_dist.shape=(576*4, 16), target.shape=(576, 4)
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        loss = []
        ch = [pred.shape[-1] // 4 for pred in pred_dist]
        for pred, tg, c in zip(pred_dist, target, ch):
            pred = pred.view(-1, c)
            tl = tg.long()  # target left
            tr = tl + 1  # target right
            wl = tr - tg  # weight left
            wr = 1 - wl  # weight right
            loss.append(F.cross_entropy(pred, tl.view(-1), reduction="none").view(tl.shape) * wl + F.cross_entropy(pred, tr.view(-1), reduction="none").view(tl.shape) * wr)

        return torch.cat(loss, dim=0).mean(-1, keepdim=True)
    
    

    def _bce_loss(self, pred_dist, target):  # pred_dist.shape=(576*4, 16), target.shape=(576, 4)
        """
        尝试将dfl分支的损失由ce改成bce, 如上面所示, loss的比例不一定直接反应输出的比例, 改成bce可以直接让输出就是想要的比例
        """
        tl = target.long()  # target left, (576, 4)
        tr = tl + 1  # target right, (576, 4)
        wl = tr - target  # weight left, (576, 4)
        wr = 1 - wl  # weight right, (576, 4)
        tl_bce = torch.zeros(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        tr_bce = torch.zeros(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        mask_l = torch.ones(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        mask_r = torch.ones(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        tl_bce.scatter_(1, tl.view(-1, 1), 1)  # (576*4, 16)
        tr_bce.scatter_(1, tr.view(-1, 1), 1)  # (576*4, 16)
        mask_l.scatter_(1, tr.view(-1, 1), 2.).type(target.dtype)  # (576*4, 16)
        mask_r.scatter_(1, tr.view(-1, 1), 2.).type(target.dtype)  # (576*4, 16)
        combined_mask = mask_l.scatter_(1, tr.view(-1, 1), 2.).type(target.dtype)  # (576*4, 16)
        tl_bce *= wl.view(-1, 1)
        tr_bce *= wr.view(-1, 1)
        bce_target = tl_bce + tr_bce   # (576*4, 16)

        # loss = self.bce(pred_dist, tl_bce.to(target.dtype)) * mask_l + self.bce(pred_dist, tr_bce.to(target.dtype)) * mask_r
        # loss = self.bce(pred_dist, tl_bce.to(target.dtype)) + self.bce(pred_dist, tr_bce.to(target.dtype))
        # loss = self.bce(pred_dist, bce_target.to(target.dtype))  # (576*4, 16)
        loss = self.bce(pred_dist, bce_target.to(target.dtype)) * combined_mask  # 给正样本位置乘以权重

        return loss.mean(-1).view(target.shape).mean(-1, keepdim=True)  # (576, 1)

        # return (self.bce(pred_dist, tl_bce.to(target.dtype)) + self.bce(pred_dist, tr_bce.to(target.dtype))).mean(-1).view(target.shape).mean(-1, keepdim=True)  # (576, 1)
        # return self.bce(pred_dist, tl_bce.to(target.dtype)) + self.bce(pred_dist, tr_bce.to(target.dtype))
    

    '''计算两个交叉熵loss, 一个是pixel的, 一个是section的'''
    def _new_loss(self, pred_dist, pred_section, target, pixel_section=None, epoch=0):  # pred_dist.shape=(576*4, 16)
        """
        尝试将dfl分支的损失由ce改成bce, 如上面所示, loss的比例不一定直接反应输出的比例, 改成bce可以直接让输出就是想要的比例
        """
        '''gfl loss for pixel prediction'''
        tl = target.long()  # target left, (576, 4)
        tr = tl + 1  # target right, (576, 4)
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        loss_dfl = (    # F.cross_entropy(pred_dist, tl.view(-1), reduction="none").shape=(pred_dist.shape[0])
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl  # shape=target.shape=(576, 4)
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)  * wr
        ).mean(-1, keepdim=True)

        '''ce loss for pixel prediction'''
        # tl = target.long()  # target left, (576, 4)
        # tr = tl + 1  # target right, (576, 4)
        # loss_dfl = (    # F.cross_entropy(pred_dist, tl.view(-1), reduction="none").shape=(pred_dist.shape[0])
        #     F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)   # shape=target.shape=(576, 4)
        #     + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)  # 这个loss不乘以权重，因为只需要得到pixel
        # ).mean(-1, keepdim=True)
        
        '''bce loss for pixel prediction'''
        # tl_bce = torch.zeros(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        # tr_bce = torch.zeros(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        # tl_bce.scatter_(1, tl.view(-1, 1), 1)  # (576*4, 16)
        # tr_bce.scatter_(1, tr.view(-1, 1), 1)  # (576*4, 16)
        # bce_target = tl_bce + tr_bce   # (576*4, 16)
        # loss_dfl = self.bce(pred_dist, bce_target.to(target.dtype)).mean(-1).view(target.shape).mean(-1, keepdim=True)

 
        # loss_dfl = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape).mean(-1, keepdim=True)  # 单one-hot

        section_target = (target - target.floor()) * (pixel_section - 1)  # shape=(576, 4)
        tl = section_target.long()  # shape=(576, 4)
        tr = tl + 1
        wl = tr - section_target
        wr = 1 - wl

        loss_section = ( 
            F.cross_entropy(pred_section, tl.view(-1), reduction="none").view(tl.shape) * wl   # shape=target.shape=(576, 4)
            + F.cross_entropy(pred_section, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

        # loss = loss_section + loss_dfl

        # tl_bce = torch.zeros(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        # tr_bce = torch.zeros(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        # mask_l = torch.ones(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        # mask_r = torch.ones(pred_dist.shape, dtype=target.dtype).cuda()   # (576*4, 16)
        # tl_bce.scatter_(1, tl.view(-1, 1), 1)  # (576*4, 16)
        # tr_bce.scatter_(1, tr.view(-1, 1), 1)  # (576*4, 16)
        # mask_l.scatter_(1, tr.view(-1, 1), 2.).type(target.dtype)  # (576*4, 16)
        # mask_r.scatter_(1, tr.view(-1, 1), 2.).type(target.dtype)  # (576*4, 16)
        # combined_mask = mask_l.scatter_(1, tr.view(-1, 1), 2.).type(target.dtype)  # (576*4, 16)
        # tl_bce *= wl.view(-1, 1)
        # tr_bce *= wr.view(-1, 1)
        # bce_target = tl_bce + tr_bce   # (576*4, 16)

        # loss = self.bce(pred_dist, tl_bce.to(target.dtype)) * mask_l + self.bce(pred_dist, tr_bce.to(target.dtype)) * mask_r
        # loss = self.bce(pred_dist, tl_bce.to(target.dtype)) + self.bce(pred_dist, tr_bce.to(target.dtype))
        # loss = self.bce(pred_dist, bce_target.to(target.dtype))  # (576*4, 16)
        # loss = self.bce(pred_dist, bce_target.to(target.dtype)) * combined_mask  # 给正样本位置乘以权重

        # weight = 0.9 - 0.4 * epoch / 100
        # loss = weight * loss_dfl + (1 - weight) * loss_section

        loss = 0.9 * loss_dfl + 0.1 * loss_section

        return loss
    


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.cv1 = True if hasattr(m, 'cv1') else False
        self.new_dfl = m.new_dfl if hasattr(m, 'new_dfl') else None
        self.pixel_section = m.pixel_section if hasattr(m, 'pixel_section') else None
        self.different_channel = m.different_channel if hasattr(m, 'different_channel') else None
        self.small_object = m.small_object if hasattr(m, 'small_object') else None
        self.xywh = m.xywh if hasattr(m, 'xywh') else None
        self.reg_loss = m.reg_loss if hasattr(m, 'reg_loss') else None
        self.iou_weight = m.iou_weight if hasattr(m, 'iou_weight') else None
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.topk = 10 + (len(self.stride) - 3) * 3

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""  
        # if preds[0].shape[1] != preds[1].shape[1] or (len(preds) ==  2 and preds[1][0].shape[1] != preds[1][1].shape[1]):
        if self.different_channel:
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            feats = preds[1] if isinstance(preds, tuple) else preds
            c1, c2, c3 = feats[0].shape[1] - self.nc, feats[1].shape[1] - self.nc, feats[2].shape[1] - self.nc

            pred_scores = torch.cat([xi[:, -self.nc:].view(feats[0].shape[0], self.nc, -1) for xi in feats], 2)   # (batch, 80, 8400)
            pred_distri = [feats[0][:, :c1].view(feats[0].shape[0], c1, -1), feats[1][:, :c2].view(feats[0].shape[0], c2, -1), feats[2][:, :c3].view(feats[0].shape[0], c3, -1)]

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (batch, 8400, 80)
            pred_distri = [x.permute(0, 2, 1).contiguous() for x in pred_distri]

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
            anchor_points, stride_tensor = make_anchors(feats, self.stride[:len(feats)], 0.5)  # (8400, 2), 每个相邻的像素相隔0.5, 第一个点为（0.5，0.5）

            # Targets
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # (num_boxes_per_batch, 6), 6: batch_id, cls, xywh, xywh is normalized
            if self.small_object:
                classes = torch.tensor([29, 30, 39, 73, 9, 15]).cuda()
                mask = torch.isin(targets[:, 1].int(), classes)  # targets.shape=(712, 6), targets[:, 1] is category, mask.shape
                # targets[mask] = targets[mask][:, -2:] * 2
                targets[mask, -2:] *= 1.5
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # shape=(batch, max_box_num, 5), 在里面xywh->xyxy, 并转成原图大小
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy, (batch, max_box_num, 1), (batch, max_box_num, 5)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # (batch, max_box_num, 1)

            # Pboxes
            if not self.new_dfl:
                pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (batch, 8400, 4), 可能的最大值为16+79.5，16为dfl的参数，79.5为8x特征图的最大的anchor中心点的位置
            elif self.pixel_section:
                pred_bboxes = self_decode_3(pred_distri, anchor_points, self.pixel_section, batch['_epoch'])
            else:
                pred_bboxes = self_decode_2(pred_distri, anchor_points)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(),   # (batch, 8400, cls_num) 
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # (batch, 8400, 4)
                anchor_points * stride_tensor,   # (8400, 2)
                gt_labels,   # (batch, max_gt_num, 1)
                gt_bboxes,   # (batch, max_gt_num, 4)
                mask_gt,     # # (batch, max_gt_num, 1), True or False for the max_gt_num dim
            )

            target_scores_sum = max(target_scores.sum(), 1)   # 286.26

            # Cls loss
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # (batch, 8400, 80)

            # Bbox loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, self.new_dfl, self.pixel_section
                )

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

        elif not self.cv1:  # 修改后的head多一个cv1分支，这里表示如果没有，则用原来的loss
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            feats = preds[1] if isinstance(preds, tuple) else preds
            pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(   # (batch, 64, 8400)
                (self.no-self.nc, self.nc), 1
            )

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (batch, 8400, 80)
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (batch, 8400, 64)

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
            anchor_points, stride_tensor = make_anchors(feats, self.stride[:len(feats)], 0.5)  # (8400, 2), 每个相邻的像素相隔0.5, 第一个点为（0.5，0.5）

            # Targets
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # (num_boxes_per_batch, 6), 6: batch_id, cls, xywh, xywh is normalized
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # shape=(batch, max_box_num, 5), 在里面xywh->xyxy, 并转成原图大小
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy, (batch, max_box_num, 1), (batch, max_box_num, 5)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # (batch, max_box_num, 1)

            # Pboxes
            if not self.new_dfl:
                pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (batch, 8400, 4), 可能的最大值为16+79.5，16为dfl的参数，79.5为8x特征图的最大的anchor中心点的位置
            elif self.pixel_section:
                pred_bboxes = self_decode_3(pred_distri, anchor_points, self.pixel_section, batch['_epoch'])
            elif self.xywh:
                pred_bboxes = self_decode_2(pred_distri, anchor_points, self.xywh)
            # elif self.reg_loss:
            #     pred_bboxes = self_decode_2(pred_distri, anchor_points, reg_loss=True)
            else:
                pred_bboxes = self_decode_2(pred_distri, anchor_points)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(),   # (batch, 8400, cls_num) 
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # (batch, 8400, 4)
                anchor_points * stride_tensor,   # (8400, 2)
                gt_labels,   # (batch, max_gt_num, 1)
                gt_bboxes,   # (batch, max_gt_num, 4)
                mask_gt,     # # (batch, max_gt_num, 1), True or False for the max_gt_num dim
            )

            target_scores_sum = max(target_scores.sum(), 1)   # 286.26

            # Cls loss
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # (batch, 8400, 80)

            # Bbox loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor
                if self.reg_loss:
                    loss[0], loss[2]= self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, self.new_dfl, 
                    self.pixel_section, self.xywh, reg_loss=True
                )
                else:
                    loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, self.new_dfl, self.pixel_section, self.xywh,
                    iou_weight=self.iou_weight
                )

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

        
        else:   # self.cv1为True
            '''改了head后的loss计算'''
            loss = torch.zeros(3, device=self.device)  # box, cls, dfl
            feats = preds[1] if isinstance(preds, tuple) else preds
            pred_offset, pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
                (4, self.reg_max * 4, self.nc), 1
            )

            pred_offset = pred_offset.permute(0, 2, 1).contiguous()  # (batch, 8400, 4)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (batch, 8400, 80)
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (batch, 8400, 64)

            dtype = pred_scores.dtype
            batch_size = pred_scores.shape[0]
            imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
            anchor_points, stride_tensor = make_anchors(feats, self.stride[:len(feats)], 0.5)  # (8400, 2), 每个相邻的像素相隔0.5, 第一个点为（0.5，0.5）

            # Targets
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)  # (num_boxes_per_batch, 6), 6: batch_id, cls, xywh, xywh is normalized
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # shape=(batch, max_box_num, 5), 在里面xywh->xyxy, 并转成原图大小
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy, (batch, max_box_num, 1), (batch, max_box_num, 5)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # (batch, max_box_num, 1)

            # Pboxes
            pred_bboxes, pred_anchor = self_decode(pred_distri, pred_offset, anchor_points, self.new_dfl)

            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(),   # (batch, 8400, cls_num) 
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # (batch, 8400, 4)
                anchor_points * stride_tensor,   # (8400, 2)
                gt_labels,   # (batch, max_gt_num, 1)
                gt_bboxes,   # (batch, max_gt_num, 4)
                mask_gt,     # # (batch, max_gt_num, 1), True or False for the max_gt_num dim
            )

            target_scores_sum = max(target_scores.sum(), 1)

            # Cls loss
            # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

            # Bbox loss
            if fg_mask.sum():
                target_bboxes /= stride_tensor
                loss[0], loss[2] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, self.new_dfl
                )

            loss[0] *= self.hyp.box  # box gain
            loss[1] *= self.hyp.cls  # cls gain
            loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


def decode(preds, anchors):
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
    pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes   # 返回(batch, 8400, 4)，xyxy的形式，大小为特征图大小


# 模型的conv输出得到原图大小的输出(*stride后), 输出传入assigner去分配正负样本
# (batch, 8400, 64), (batch, 8400, 4), (8400, 2)
def self_decode(pred_dist, pred_offset, anchor_points, dfl=True):    
    b, a, c = pred_dist.shape  # batch, anchors, channels

    if dfl:
        proj = torch.arange(16, dtype=torch.float, device=pred_dist.device)
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    else:
        # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).max(dim=-1)[1].type(pred_dist.dtype)  # shape=(batch, 8400, 4), 得到坐标后dtype为int64, 转回原来的float16
        '''对(4, 8400, 4, 16), 下面取16这个维度相邻最大的两个相邻值的加权平均为输出, 即和dfl的loss计算对应, 得到(4, 8400, 4)'''
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).type(pred_dist.dtype)
        pred_dist = process_tensor(pred_dist)
    lt, rb = pred_dist.chunk(2, dim=-1)
    x1y1 = anchor_points - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
    x2y2 = anchor_points + rb

    pred_anchor = torch.cat((x1y1, x2y2), dim=-1)   # (batch, 8400, 4), xyxy, 大小为特征图大小

    # (batch, 8400, 4), (batch, 8400, 4)
    return decode(pred_offset, pred_anchor), pred_anchor



def process_tensor(pred_dist, xywh=None):   # NOTE xywh值的是不是用ltrb来回归box，而是这里用得到ltrb的方法得到wh，xy由别处的sigmoid得到
    if xywh:    # pred_dist=(batch, 8400, 2, 31)
        adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 2, 30)

        # Step 2: Find the indices of the maximum adjacent sum
        max_sum_indices = torch.argmax(adjacent_sums, dim=-1)  # Shape: (4, 8400, 2)

        # Step 3: Gather the two adjacent values corresponding to max_sum_indices
        indices1 = max_sum_indices   # (4, 8400, 2)
        indices2 = indices1 + 1
        gathered_values1 = torch.gather(pred_dist, dim=-1, index=indices1.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 2)
        gathered_values2 = torch.gather(pred_dist, dim=-1, index=indices2.unsqueeze(-1)).squeeze(-1)  # Shape: (4, 8400, 2)
        summed_values = gathered_values1 + gathered_values2

        weights1 = gathered_values1 / summed_values  # Shape: (4, 8400, 2)
        weights2 = gathered_values2 / summed_values  # Shape: (4, 8400, 2)

        final_tensor = indices1 * weights1 + indices2 * weights2  # Shape: (4, 8400, 2)

    elif pred_dist.shape[-1] == 16:
        # Step 1: Get the sum of adjacent values in the 8 dimension
        adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 4, 15)

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


def self_decode_2(pred_dist, anchor_points, xywh=None):
    if xywh:
        b, a, c = pred_dist.shape  # batch, anchors, channels
        xy = pred_dist[..., :2]   # (batch, 8400, 2)
        wh = pred_dist[..., 2:]   # (batch, 8400, 62)
        pred_xy = xy.sigmoid() * 2 - 1 + anchor_points  # (batch, 8400, 2)
        pred_wh = wh.view(b, a, 2, wh.shape[-1] // 2).softmax(3).type(pred_dist.dtype)  # softmax values, (batch, 8400, 2, 31)
        pred_wh = process_tensor(pred_wh, xywh)   # (batch, 8400, 2)
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)  # (batch, 8400, 4)
        pred_xyxy = xywh2xyxy(pred_xywh)
    elif isinstance(pred_dist, list):
        feat1 = pred_dist[0].view(pred_dist[0].shape[0], pred_dist[0].shape[1], 4, pred_dist[0].shape[2] // 4).softmax(3).type(pred_dist[0].dtype)  # (b, 6400, 4, 13)
        feat2 = pred_dist[1].view(pred_dist[1].shape[0], pred_dist[1].shape[1], 4, pred_dist[1].shape[2] // 4).softmax(3).type(pred_dist[0].dtype)  # (b, 1600, 4, 16)
        feat3 = pred_dist[2].view(pred_dist[2].shape[0], pred_dist[2].shape[1], 4, pred_dist[2].shape[2] // 4).softmax(3).type(pred_dist[0].dtype)  # (b, 400, 4, 31)
        out1 = process_tensor(feat1)
        out2 = process_tensor(feat2)
        out3 = process_tensor(feat3)
        pred_dist = torch.cat([out1, out2, out3], dim=1)
        lt, rb = pred_dist.chunk(2, dim=-1)
        x1y1 = anchor_points - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
        x2y2 = anchor_points + rb
        pred_xyxy = torch.cat((x1y1, x2y2), dim=-1)   # (batch, 8400, 4), xyxy, 大小为特征图大小
    # elif reg_loss:    
    #     b, a, c = pred_dist.shape  # batch, anchors, channels
    #     # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).max(dim=-1)[1].type(pred_dist.dtype)  # shape=(batch, 8400, 4), 得到坐标后dtype为int64, 转回原来的float16
    #     '''对(4, 8400, 4, 16), 下面取16这个维度相邻最大的两个相邻值的加权平均为输出, 即和dfl的loss计算对应, 得到(4, 8400, 4)'''
    #     pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).type(pred_dist.dtype)  # softmax values, (batch, 8400, 4, 16)
    #     # pred_dist = torch.sigmoid(pred_dist.view(b, a, 4, c // 4)).type(pred_dist.dtype)  # sigmoid values, (batch, 8400, 4, 16)
    #     pred_dist, w1, w2 = process_tensor(pred_dist)  # in = (batch, 8400, 4, 16), out = (batch, 8400, 4), 4 is ltrb
    #     lt, rb = pred_dist.chunk(2, dim=-1)
    #     x1y1 = anchor_points - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
    #     x2y2 = anchor_points + rb
    #     pred_xyxy = torch.cat((x1y1, x2y2), dim=-1)   # (batch, 8400, 4), xyxy, 大小为特征图大小
    #     return pred_xyxy, w1, w2
    else:    
        b, a, c = pred_dist.shape  # batch, anchors, channels
        # pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).max(dim=-1)[1].type(pred_dist.dtype)  # shape=(batch, 8400, 4), 得到坐标后dtype为int64, 转回原来的float16
        '''对(4, 8400, 4, 16), 下面取16这个维度相邻最大的两个相邻值的加权平均为输出, 即和dfl的loss计算对应, 得到(4, 8400, 4)'''
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).type(pred_dist.dtype)  # softmax values, (batch, 8400, 4, 16)
        # pred_dist = torch.sigmoid(pred_dist.view(b, a, 4, c // 4)).type(pred_dist.dtype)  # sigmoid values, (batch, 8400, 4, 16)
        pred_dist = process_tensor(pred_dist)  # in = (batch, 8400, 4, 16), out = (batch, 8400, 4), 4 is ltrb
        lt, rb = pred_dist.chunk(2, dim=-1)
        x1y1 = anchor_points - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
        x2y2 = anchor_points + rb
        pred_xyxy = torch.cat((x1y1, x2y2), dim=-1)   # (batch, 8400, 4), xyxy, 大小为特征图大小

    # (batch, 8400, 4), (batch, 8400, 4)
    return pred_xyxy


def self_decode_3(pred_dist, anchor_points, pixel_section, epoch):    
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
    pred_dist = process_tensor_2(dfl_prediction, section_prediction, pixel_section, epoch)  # (batch, 8400, 4)
    lt, rb = pred_dist.chunk(2, dim=-1)
    x1y1 = anchor_points - lt   # anchor_points为特征图大小，即8x上的max为79.5, 16x上的max为39.5
    x2y2 = anchor_points + rb

    pred_anchor = torch.cat((x1y1, x2y2), dim=-1)   # (batch, 8400, 4), xyxy, 大小为特征图大小

    # (batch, 8400, 4), (batch, 8400, 4)
    return pred_anchor


def process_tensor_2(pred_dist, section_prediction, pixel_section, epoch):
    # Step 1: Get the sum of adjacent values in the 8 dimension
    adjacent_sums = pred_dist[..., :-1] + pred_dist[..., 1:]  # Shape: (4, 8400, 4, 15)
    adjacent_sums_2 = section_prediction[..., :-1] + section_prediction[..., 1:]  # Shape: (4, 8400, 4, 9)

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

    # weight = 0.9 - 0.4 * epoch / 100   # 100个epoch, 0.9 -> 0.5

    # output = weight * output_2 + (1 - weight) * output

    output = 0.9 * output_2 + 0.1 * output

    return output