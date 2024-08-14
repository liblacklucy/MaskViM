# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from argparse import ArgumentParser
from typing import Type, Sequence, Union

import os
import os.path as osp
from tqdm import tqdm

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np

from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmengine.runner import Runner
from mmengine.runner.loops import _InfiniteDataloaderIterator


from mmseg.apis import inference_model, init_model, utils
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules, SampleList
from mmseg.visualization import SegLocalVisualizer
from mmseg.models import BaseSegmentor

# import model
from mods.backbone.exp import *
from mods.dataloader.Mapillary_19 import *
from mods.dataloader.GTAV_19 import *
from mods.dataloader.Synthia_19 import *
from mods.hooks.exp import *


class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        # self.data_buffer.clear()
        pass


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    runner = Runner.from_cfg(cfg)

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)
    # exit(1)

    source = [
        # 'decode_head.fusion.stages.0.query_project.activate',
        # 'decode_head.context.stages.0.key_project.activate',
        # 'decode_head.fpn_bottleneck.conv',  # 该层特征仅，显存允许保存280个样本
        # 'decode_head.bottleneck.conv',  # 经过psp的最后一层特征
        # 'decode_head.bottleneck.bn',  # 经过psp的最后一层特征
        # 'decode_head.bottleneck.activate',  # 经过psp的最后一层特征
        # 'decode_head.fpn_bottleneck.bn',
        'decode_head.fpn_bottleneck.activate',  # 分类层前一层输出特征
        # 'decode_head.conv_seg',
        # 'backbone.outnorm3'
    ]  # set len(source) == 1
    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    label_lst = []
    # innerfeat_lst = []
    data_loader = runner.test_dataloader
    # data_loader = runner.train_dataloader  # 设置DefaultSampler和batchsize=1
    for idx, data in tqdm(enumerate(data_loader)):  # batch is 1
        # print(idx)
        if idx == 250:
            break
        with torch.no_grad():
            img = torch.stack(data['inputs'], dim=0).float().cuda()
            label = data['data_samples'][0].gt_sem_seg.data.cuda()
            # print(len(data['data_samples']))
            # print(img.shape)
            # print(label.shape)
            result = model(img)
            label_lst.append(label)

    # with recorder:
    #     assert len(recorder.data_buffer) == count * len(label_lst), 'error inner_feat count !'
    #     for idx, feature in tqdm(enumerate(recorder.data_buffer)):
    #         print(feature.shape)
    #         innerfeat_lst.append(feature)
    with recorder:
        assert len(recorder.data_buffer) == len(label_lst), 'error inner_feat count !'
        # torch.save(recorder.data_buffer, '/data/ljh/data/MaskViM/city/innerfeat_lst.pth')
        torch.save(recorder.data_buffer, os.path.join(cfg.work_dir, 'innerfeat_lst.pth'))

    torch.save(label_lst, os.path.join(cfg.work_dir, 'innerfeat_label_lst.pth'))
    # torch.save(innerfeat_lst, '/data/ljh/data/MaskViM/city/innerfeat_lst.pth')
    print('done')


if __name__ == '__main__':
    main()
