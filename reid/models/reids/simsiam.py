import torch
import torch.nn.functional as F

from reid.utils import concat_all_gather
from ..builder import REIDS
from .baseline import Baseline


@REIDS.register_module()
class SimSiam(Baseline):

    def set_epoch(self, epoch, **kwargs):
        self._epoch = epoch

    def cascade(self, z1, z2):
        targets = z1.new_zeros(z1.size())

        z1 = z1.clone().detach()
        z2 = z2.clone().detach()
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z2 = concat_all_gather(z2)
        num = self._epoch // 10
        sim = torch.matmul(z1, z2.t())
        _, inds_sorted = torch.sort(sim, dim=1, descending=True)

        for i in range(z1.shape[0]):
            targets[i] = torch.mean(
                z2[inds_sorted[i, :num]], dim=0, keepdim=True)

        return targets

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, f'img must be 5 dims, but got: {img.dim()}'
        N, _, C, H, W = img.shape

        img = img.reshape(N * 2, C, H, W)
        z = self.neck(self.backbone(img))[0]

        z1, z2 = torch.unbind(z.reshape(N, 2, -1), dim=1)
        if self._epoch // 10 == 0:
            loss = self.head(z1, z2)['loss'] + self.head(z2, z1)['loss']
        else:
            z1_targets = self.cascade(z1, z2)
            z2_targets = self.cascade(z2, z1)
            loss = self.head(z1, z1_targets)['loss'] + self.head(
                z2, z2_targets)['loss']

        return dict(loss=loss)
