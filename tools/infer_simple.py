#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--inputlist',
        type=str
    )
    parser.add_argument(
        '--basepath',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    # comment old file
    # if os.path.isdir(args.im_or_folder):
        # im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    # else:
        # im_list = [args.im_or_folder]
    # use inputlist
    if args.inputlist:
        # print(">>> args.inputlist") 
        def read_file(fname):
            with open(fname) as f:
                content = f.readlines()
                content = [x.strip() for x in content]
            return content
        imgnames = read_file(args.inputlist)
        # print(type(imgnames))
        # print(imgnames)
        im_list = imgnames
    # add base path
    if args.basepath:
        im_list = [os.path.join(args.basepath, x) for x in im_list]

    result_all = []
    for i, im_name in enumerate(im_list):
        print(">>> im_name", im_name)
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        if not os.path.exists(im_name):
            raise ValueError("image name not exists,", im_name)
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        import numpy as np
        np.set_printoptions(suppress=True)
        # print(type(cls_keyps))
        # print("len", len(cls_keyps))
        # print(type(cls_keyps[0]))
        # print("cls_keyps[0] len", len(cls_keyps[0]))
        # print("cls_keyps[1] len", len(cls_keyps[1]))
        cls_keyps1 = cls_keyps[1]
        cls_keyps1 = np.array(cls_keyps1) 
        pts = cls_keyps1[0][:4, :]
        pts = np.transpose(pts, (1, 0))
        pts_lt = pts.tolist()
        # print("pts_lt", pts_lt)
        # print("pts.shape", pts.shape)
        result_all.append(
                {
                    "image_id": im_name,
                    "keypoint": pts_lt
                    }
                )
        """
        n x 4 x 17
        """
        # cls_keyps1 = np.reshape(cls_keyps1, (-1,3))
        # print("cls_keyps1.reshape", cls_keyps1.shape)
        # print("one.shape", cls_keyps1[0].shape)
        # print("cls_keyps1", cls_keyps1[0])
        # print("type cls_keyps1", type(cls_keyps1))
        # print(cls_keyps1)

        # vis_utils.vis_one_image(
            # im[:, :, ::-1],  # BGR -> RGB for visualization
            # im_name,
            # args.output_dir,
            # cls_boxes,
            # cls_segms,
            # cls_keyps,
            # dataset=dummy_coco_dataset,
            # box_alpha=0.3,
            # show_class=True,
            # thresh=args.thresh,
            # kp_thresh=args.kp_thresh,
            # ext=args.output_ext,
            # out_when_no_box=args.out_when_no_box
        # )

    import json
    def write_json(fname, data):
        with open(fname, 'w') as outfile:
            json.dump(data, outfile) 
    write_json("maskrcnn_h36m.json", result_all)
    



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
