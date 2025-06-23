import torch
import torch.nn.functional as F
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

@MODELS.register_module()
class EGAFDeepSupHead(BaseDecodeHead):
    """Decode head that直接接收 EGAF 输出的 5 个 logits.

    Args:
        loss_weights (Sequence[float]):   主输出及 4 个辅助输出的权重.
        main_index (int):  推理阶段选用哪一路作为最终结果.
    """
    def __init__(self,
                 in_channels=(5, 5, 5, 5, 5),   # 每路 logits 的通道数
                 in_index=(0, 1, 2, 3, 4),
                 loss_weights=(1., 0.4, 0.3, 0.2, 0.2),
                 main_index=0,
                 **kwargs):

        super().__init__(           # BaseDecodeHead 需要传 channels 等
            in_channels=in_channels,
            in_index=in_index,
            channels=in_channels[0],    # 随便填，反正不卷积
            input_transform='multiple_select',
            **kwargs)

        assert len(loss_weights) == len(in_index)
        self.loss_weights = loss_weights
        self.main_index = main_index

    # ------------------------------------------------ #
    # 推理阶段仅返回 main_index 的 logits
    # ------------------------------------------------ #
    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)  # list -> list(5)

        if self.training:  # 训练阶段 → 要 5 路一起算深监督
            return tuple(inputs)  # 让 loss_by_feat() 拿到 5 个 logits
        else:  # 推理 / 验证阶段 → 只需一路结果
            return inputs[self.main_index]

    # ------------------------------------------------ #
    # 训练阶段：5 路 logits 各算 CE
    # ------------------------------------------------ #
    def loss_by_feat(self, seg_logits, batch_data_samples):
        if not isinstance(seg_logits, (list, tuple)):
            raise TypeError('EGAFDeepSupHead expects tuple of 5 logits')

        seg_label = self._stack_batch_gt(batch_data_samples)  # (N,1,H,W)
        if seg_label.ndim == 4 and seg_label.shape[1] == 1:  # <-- 新增
            seg_label = seg_label.squeeze(1)  # (N,H,W)

        losses, total = {}, 0.
        for i, logit in enumerate(seg_logits):
            if logit.shape[-2:] != seg_label.shape[-2:]:
                logit = F.interpolate(
                    logit, size=seg_label.shape[-2:], mode='bilinear',
                    align_corners=self.align_corners)

            loss_i = self.loss_decode(
                logit, seg_label, ignore_index=self.ignore_index)
            losses[f'loss_seg_{i}'] = loss_i * self.loss_weights[i]
            total += losses[f'loss_seg_{i}']

        losses['loss_seg'] = total
        return losses