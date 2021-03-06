import os.path as osp

import torch

from .builder import PIPELINES


@PIPELINES.register_module()
class RandomCamStyle(object):
    """Randomly apply CamStyle.
    """

    def __init__(self,
                 dataset,
                 camstyle_root='bounding_box_train_camstyle',
                 p=0.5):
        self.dataset = dataset
        self.data_source = dataset.data_source

        self.data_root = self.data_source.data_root
        self.camstyle_root = osp.join(self.data_root, camstyle_root)
        self.num_cams = self.data_source.NUM_CAMERAS
        self.p = p

    def _random_camid(self, camid):
        while True:
            rand_camid = torch.randperm(self.num_cams)[0].item() + 1
            if rand_camid != camid:
                return rand_camid

    def __call__(self, img, camid):
        if self.p < torch.rand(1):
            return img

        rand_camid = self._random_camid(camid)
        img = osp.basename(img)[:-4]  # remove postfix
        if self.data_source.DATA_SOURCE == 'MSMT17':
            img = f'{img}_fake_{rand_camid}.jpg'
        else:
            img = f'{img}_fake_{camid}to{rand_camid}.jpg'
        img = osp.join(self.camstyle_root, img)

        return img
