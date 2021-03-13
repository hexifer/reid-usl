import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS, build_neck


@HEADS.register_module()
class LatentPredictHead(nn.Module):

    def __init__(self, predictor, **kwargs):
        super(LatentPredictHead, self).__init__()
        self.predictor = build_neck(predictor)

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (Tensor): NxC input features.
            targets (Tensor): NxC input features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([inputs])[0]
        loss = -F.cosine_similarity(
            pred, targets.clone().detach(), dim=-1).mean()

        return dict(loss=loss)
