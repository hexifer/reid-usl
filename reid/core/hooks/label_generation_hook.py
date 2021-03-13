import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook

from ..label import build_label_generator
from .extractor import Extractor


@HOOKS.register_module()
class LabelGenerationHook(Hook):

    def __init__(self, extractor, label_generator, start=1, interval=1):
        self.extractor = Extractor(**extractor)
        self.label_generator = build_label_generator(label_generator)
        self.start = start
        self.interval = interval

        self.distributed = dist.is_available() and dist.is_initialized()

    @torch.no_grad()
    def _dist_gen_labels(self, feats):
        if dist.get_rank() == 0:
            labels = self.label_generator.gen_labels(feats)[0]
            labels = labels.cuda()
        else:
            labels = torch.zeros(feats.shape[0], dtype=torch.long).cuda()
        dist.broadcast(labels, 0)

        return labels

    @torch.no_grad()
    def _non_dist_gen_labels(self, feats):
        labels = self.label_generator.gen_labels(feats)[0]

        return labels.cuda()

    def gen_flag(self, runner):
        if self.start is None:
            if not self.every_n_epochs(runner, self.interval):
                return False  # No evaluation
        elif (runner.epoch + 1) < self.start:
            # No evaluation if start is larger than the current epoch.
            return False
        else:
            # Evaluation only at epochs 3, 5, 7... if start==3 and interval==2
            if (runner.epoch + 1 - self.start) % self.interval:
                return False
        return True

    def before_train_epoch(self, runner):
        if not self.gen_flag(runner):
            return

        with torch.no_grad():
            feats = self.extractor.extract_feats(runner.model)
            if self.distributed:
                labels = self._dist_gen_labels(feats)
            else:
                labels = self._non_dist_gen_labels(feats)

        runner.model.train()

        labels = labels.cpu().numpy()
        runner.data_loader.dataset.update_labels(labels)
        if hasattr(runner.data_loader.sampler, 'init_data'):
            # identity sampler cases
            runner.data_loader.sampler.init_data()

        self.evaluate(runner, labels)

    def evaluate(self, runner, labels):
        hist = np.bincount(labels)
        clusters = np.where(hist > 1)[0]
        unclusters = np.where(hist == 1)[0]
        runner.logger.info(f'{self.__class__.__name__}: '
                           f'{clusters.shape[0]} clusters, '
                           f'{unclusters.shape[0]} unclusters')
