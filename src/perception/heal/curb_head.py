"""Single-frame curb head on HEAL HENet P8/P16 features.

This module defines the float training graph and bounded candidate output.
OE prepare/export belongs after these structural changes; importing this
module does not substitute for a Horizon Plugin integration check.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CurbHead(nn.Module):
    def __init__(self, channels=(128, 192), width=64, *, auxiliary_region=True,
                 multiscale=True, with_offset=True):
        super().__init__()
        self.config = dict(auxiliary_region=auxiliary_region, multiscale=multiscale, with_offset=with_offset)
        self.p8 = nn.Conv2d(channels[0], width, 1)
        self.p16 = nn.Conv2d(channels[1], width, 1) if multiscale else None
        layers = []
        for _ in range(2):
            layers.extend((nn.Conv2d(width, width, 3, padding=1, groups=width),
                           nn.ReLU(), nn.Conv2d(width, width, 1), nn.ReLU()))
        self.refine = nn.Sequential(*layers)
        self.heatmap = nn.Conv2d(width, 1, 1)
        self.offset = nn.Conv2d(width, 2, 1) if with_offset else None
        self.region = nn.Conv2d(width, 4, 1) if auxiliary_region else None
        nn.init.constant_(self.heatmap.bias, math.log(.01 / .99))

    def forward(self, p8, p16):
        x = self.p8(p8)
        if self.p16 is not None:
            x = x + F.interpolate(self.p16(p16), size=p8.shape[-2:], mode='bilinear', align_corners=False)
        x = F.interpolate(self.refine(x), scale_factor=2, mode='bilinear', align_corners=False)
        heatmap = self.heatmap(x)
        offsets = .5 * torch.tanh(self.offset(x)) if self.offset is not None else torch.zeros_like(heatmap).expand(-1, 2, -1, -1)
        result = {'heatmap': heatmap, 'offset': offsets}
        if self.training and self.region is not None:
            result['region'] = self.region(x)
        return result


class CurbCandidates(nn.Module):
    """At most K candidates per view, in input-image pixels (x, y).

    Offsets are relative to stride-4 cell centres. Invalid padded slots have
    zero coordinates/scores. Truncation counts *all* thresholded local peaks,
    so clipping the list never implies that the whole curb was observed.
    """
    def __init__(self, k=256, threshold=.3, stride=4):
        super().__init__()
        if type(k) is not int or not 0 < k <= 256 or not 0 < threshold < 1 or stride <= 0:
            raise ValueError('invalid bounded curb candidate settings')
        self.k, self.threshold, self.stride = k, threshold, stride

    def forward(self, heatmap, offset):
        scores = heatmap.sigmoid()
        # Preserve a line's cross-section maxima along either image axis.
        # Isotropic maxima also suppress along the tangent and break smooth roots into isolated points.
        vertical = scores == F.max_pool2d(scores, (3, 1), stride=1, padding=(1, 0))
        horizontal = scores == F.max_pool2d(scores, (1, 3), stride=1, padding=(0, 1))
        peaks = (vertical | horizontal) & (scores >= self.threshold)
        ranked = torch.where(peaks, scores, torch.zeros_like(scores)).flatten(1)
        values, indices = torch.topk(ranked, self.k, dim=1, sorted=True)
        valid = values >= self.threshold
        selected_offset = torch.gather(offset.flatten(2), 2, indices[:, None].expand(-1, 2, -1))
        width = heatmap.shape[-1]
        xy = torch.stack(((indices % width).to(offset.dtype) + .5,
                          torch.div(indices, width, rounding_mode='floor').to(offset.dtype) + .5), dim=-1)
        xy = (xy + selected_offset.transpose(1, 2)) * self.stride
        return {'curb_xy': torch.where(valid[..., None], xy, torch.zeros_like(xy)),
                'curb_scores': torch.where(valid, values, torch.zeros_like(values)),
                'curb_valid': valid, 'curb_truncated': peaks.flatten(1).sum(1) > self.k}


def curb_losses(prediction, target):
    """Visible root heatmap, subpixel offsets and optional region supervision.

    ``visible`` is explicit: uncertain/occluded labels contribute no loss.
    Region labels use 255 for ignore. Inputs contain supplied labels only;
    this function neither reads CARLA truth nor manufactures root targets
    from the road/sidewalk category boundary.
    """
    logits, offsets = prediction['heatmap'], prediction['offset']
    heat = target['heatmap'].to(logits.dtype)
    visible = target['visible'].bool()
    positive = (heat == 1) & visible
    negative = (heat < 1) & visible
    probability = logits.sigmoid()
    focal = (-F.logsigmoid(logits) * (1-probability).pow(2) * positive
             - F.logsigmoid(-logits) * probability.pow(2) * (1-heat).pow(4) * negative)
    count = positive.sum().clamp_min(1)
    mask = target['offset_valid'].bool() & visible
    regression = F.smooth_l1_loss(offsets, target['offset'].to(offsets.dtype), reduction='none')
    losses = {'loss_curb_heatmap': focal.sum() / count,
              'loss_curb_offset': (regression * mask).sum() / (2 * mask.sum()).clamp_min(1)}
    if 'region' in prediction:
        labels = target['region'].long()
        labels = torch.where(visible.squeeze(1), labels, torch.full_like(labels, 255))
        region_loss = F.cross_entropy(prediction['region'], labels, ignore_index=255, reduction='none')
        losses['loss_curb_region'] = region_loss.sum() / (labels != 255).sum().clamp_min(1)
    return losses
