# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    # validate(cfg, valid_loader, valid_dataset, model, criterion,
    #          final_output_dir, tb_log_dir)

    all_detail = validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    # print(all_detail)

    # todo: 在这里进行统计预测结果 ---------------------------------
    import json
    with open("my_all_result_office.json", mode="w", encoding="utf-8") as out:
        out.write(json.dumps(all_detail, ensure_ascii=False))

    # total_num = 0  # 统计共有多少个检测出来了(有些标注文件就是全0)
    # no_detail = 0  # 记录没有标注文件
    #
    # detail_17 = 0  # 统计17关键点值
    # index_17 = 0
    # detail_18 = 0
    # index_18 = 0
    # detail_19 = 0
    # index_19 = 0
    # detail_20 = 0
    # index_20 = 0
    # detail_21 = 0
    # index_21 = 0
    # for this_detail_index, this_detail in enumerate(all_detail):
    #     if this_detail == {}:
    #         no_detail = no_detail + 1
    #     else:
    #         if "17" in this_detail.keys():
    #             detail_17 = detail_17 + this_detail["17"]
    #             index_17 = index_17 + 1
    #         if "18" in this_detail.keys():
    #             detail_18 = detail_18 + this_detail["18"]
    #             index_18 = index_18 + 1
    #         if "19" in this_detail.keys():
    #             detail_19 = detail_19 + this_detail["19"]
    #             index_19 = index_19 + 1
    #         if "20" in this_detail.keys():
    #             detail_20 = detail_20 + this_detail["20"]
    #             index_20 = index_20 + 1
    #         if "21" in this_detail.keys():
    #             detail_21 = detail_21 + this_detail["21"]
    #             index_21 = index_21 + 1
    # # 打印结果
    # print(f"关键点17的平均欧氏距离为: {(detail_17 / index_17)}\n")
    # print(f"关键点18的平均欧氏距离为: {(detail_18 / index_18)}\n")
    # print(f"关键点19的平均欧氏距离为: {(detail_19 / index_19)}\n")
    # print(f"关键点20的平均欧氏距离为: {(detail_20 / index_20)}\n")
    # print(f"关键点21的平均欧氏距离为: {(detail_21 / index_21)}\n")
    #
    # print("-------------------总的5个点平均欧式距离为-------------------\n")
    # total_oushijuli = (detail_17 + detail_18 + detail_19 + detail_20 + detail_21) / (index_17 + index_18 + index_19 + index_20 + index_21)
    # print(total_oushijuli)
    # ---------------------------------------------------------

if __name__ == '__main__':
    main()
