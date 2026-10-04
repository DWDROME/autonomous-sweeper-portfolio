"""Own-code extension of the pinned HEAL BevFusion graph, before OE prepare.

Load the original checkpoint strictly before wrapping it. The original BEV
neck and losses remain intact. Model outputs have explicit names and omit
the unused BEV embedding; no model logits are handed to a CPU reducer.
"""
import torch
from torch import nn

from .curb_head import CurbCandidates, curb_losses
from .compact_occ import CompactOccExit


class SharedPerception(nn.Module):
    def __init__(self, base, curb_head=None, *, work_views=(), candidates=256):
        super().__init__()
        if len(base.bev_decoders) != 2 or base.camera_net is None or base.lidar_net is None:
            raise ValueError('expected the pinned HEAL camera/LiDAR detection + Occ model')
        if ((curb_head is None) != (len(work_views) == 0) or len(work_views) > 2
                or len(set(work_views)) != len(work_views)
                or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 6 for v in work_views)):
            raise ValueError('curb head requires one or two distinct configured six-camera indices')
        self.base = base
        self.curb_head = curb_head
        self.work_views = tuple(work_views)
        self.semantics = CompactOccExit()
        self.candidates = CurbCandidates(candidates)
        self.dequant_curb_xy = torch.quantization.DeQuantStub()
        self.dequant_curb_scores = torch.quantization.DeQuantStub()
        self.dequant_curb_losses = nn.ModuleDict({name: torch.quantization.DeQuantStub()
            for name in ('heatmap', 'offset', 'region')})
        self._curb_base_frozen = False

    def train(self, mode=True):
        super().train(mode)
        if self._curb_base_frozen:
            self.base.eval()
        return self

    def freeze_base_for_curb(self):
        """Freeze inherited weights and BatchNorm state while training the new head."""
        if self.curb_head is None:
            raise ValueError('head-only training requires a configured curb head')
        self._curb_base_frozen = True
        self.base.requires_grad_(False)
        self.base.eval()
        self.curb_head.requires_grad_(True)
        return self

    def curb_features(self, images):
        """Extract the inference P8/P16 views with the frozen encoder."""
        if not self._curb_base_frozen:
            raise ValueError('call freeze_base_for_curb before feature extraction')
        assert images.ndim == 4 and images.shape[1] == 3 and images.shape[0] % 6 == 0
        self.base.eval()
        with torch.no_grad():
            stages = self.base.camera_net.backbone(images)
            batch = images.shape[0] // 6
            return tuple(stages[i].reshape(batch, 6, *stages[i].shape[1:])
                         [:, self.work_views].flatten(0, 1) for i in (2, 3))

    def forward_curb(self, images=None, *, cached_features=None):
        """Run the head on images or the trainer's bound feature cache."""
        if not self._curb_base_frozen:
            raise ValueError('call freeze_base_for_curb before head-only training')
        if (images is None) == (cached_features is None):
            raise ValueError('supply images or cached features exclusively')
        features = self.curb_features(images) if images is not None else cached_features
        return {name: self.dequant_curb_losses[name](value)
                for name, value in self.curb_head(*features).items()}

    def features(self, data):
        camera = self.base.camera_net
        # HENet returns [P2 (upsampled), P4, P8, P16, P32]. Read the actual
        # stage list, retaining the existing out_indices/neck for BEV.
        stages = camera.backbone(data['img'])
        camera_features = camera.neck([stages[i] for i in camera.out_indices])
        camera_bev = camera.view_transformer(camera_features, data)
        batch, _, channels = camera_bev.shape
        camera_bev = camera_bev.permute(0, 2, 1).reshape(batch, channels, self.base.bev_h, self.base.bev_w)
        lidar_bev = self.base.forward_lidar_feature(data)
        fused = self.base.fuse_module(torch.cat((camera_bev, lidar_bev), dim=1))
        curb = None
        if self.curb_head is not None:
            # Batch is [vehicle, six cameras]. Select views BEFORE projections
            # and the new decoder, never run a second image encoder.
            p8 = stages[2].reshape(batch, 6, *stages[2].shape[1:])[:, self.work_views]
            p16 = stages[3].reshape(batch, 6, *stages[3].shape[1:])[:, self.work_views]
            curb = self.curb_head(p8.flatten(0, 1), p16.flatten(0, 1))
        return fused, curb

    def forward(self, data):
        fused, curb = self.features(data)
        if self.training:
            losses = self.base.post_process(data, fused)
            if curb is not None:
                decoded = {name: self.dequant_curb_losses[name](value) for name, value in curb.items()}
                losses.update(curb_losses(decoded, data['curb_targets']))
            return losses

        detector, occupancy = self.base.bev_decoders
        _, classes, references, coordinates = detector.bev_decoder(detector.get_bev_embed(fused))
        if not detector.is_compile:
            classes, references, coordinates = classes[-1], references[-1], coordinates[-1]
        # Keep the existing ROI/head, and move the output boundary after
        # class reduction. Actual OE node assignment is checked at
        # compilation; this float graph alone does not establish BPU support.
        occ_feature = fused if occupancy.roi_resizer is None else occupancy.roi_resizer([fused])['roi_feat0']
        result = {'outputs_classes': classes, 'reference_out': references, 'outputs_coords': coordinates}
        labels, scores = self.semantics(occupancy.occ_head(occ_feature))
        result.update(occ_labels=labels, occ_scores=scores)
        if curb is not None:
            candidates = self.candidates(curb['heatmap'], curb['offset'])
            candidates['curb_xy'] = self.dequant_curb_xy(candidates['curb_xy'])
            candidates['curb_scores'] = self.dequant_curb_scores(candidates['curb_scores'])
            result.update(candidates)
        return result


def decode_detections(outputs, detector):
    """The upstream final-layer host decoder, without a BEV tensor copy."""
    coordinates, reference = outputs['outputs_coords'].float(), outputs['reference_out'].float()
    limits = detector.pc_range
    x = (coordinates[..., 0:1] + reference[..., 0:1]).sigmoid() * (limits[3]-limits[0]) + limits[0]
    y = (coordinates[..., 1:2] + reference[..., 1:2]).sigmoid() * (limits[4]-limits[1]) + limits[1]
    z = (coordinates[..., 4:5] + reference[..., 2:3]).sigmoid() * (limits[5]-limits[2]) + limits[2]
    boxes = torch.cat((x, y, coordinates[..., 2:4], z, coordinates[..., 5:]), dim=-1)
    return detector.post_process({'all_cls_scores': outputs['outputs_classes'].float().unsqueeze(0),
        'all_bbox_preds': boxes.unsqueeze(0), 'enc_cls_scores': None, 'enc_bbox_preds': None})
