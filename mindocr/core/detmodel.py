
import json
import logging
import os
import sys
from typing import List

import numpy as np
# from config import parse_args
from .postprocess import Postprocessor
from .preprocess import Preprocessor
from shapely.geometry import Polygon
from .utils import *

import mindspore as ms

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from ..models import build_model
from ..data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ..utils.logger import set_logger
from ..utils.visualize import draw_boxes, show_imgs

__all__=["DetModel"]

class DetModel(object):

    def __init__(self, det="DB++", **kwargs):
        self.image_dir = kwargs.get("image_dir")
        self.det_algorithm = kwargs.get("det_algorithm", "DB++")
        self.det_amp_level = kwargs.get("det_amp_level", "O0")
        self.det_model_dir = kwargs.get("det_model_dir", None)
        self.det_limit_side_len = kwargs.get("det_limit_side_len", 960)
        self.det_limit_type = kwargs.get("det_limit_type", "max")
        self.det_box_type = kwargs.get("det_box_type", "quad")
        self.draw_img_save_dir = kwargs.get("draw_img_save_dir", "./inference_results")
        self.visualize_preprocess = False
        self.algo_to_model_name = {
            "DB": "dbnet_resnet50",
            "DB++": "dbnetpp_resnet50",
            "DB_MV3": "dbnet_mobilenetv3",
            "DB_PPOCRv3": "dbnet_ppocrv3",
            "PSE": "psenet_resnet152",
        }
        self.model_name = self.algo_to_model_name[self.det_algorithm]

        os.makedirs(self.draw_img_save_dir, exist_ok=True)


    def preprocess_f(self):
        self.preprocess = Preprocessor(
            task="det",
            algo=self.det_algorithm,
            det_limit_side_len=self.det_limit_side_len,
            det_limit_type=self.det_limit_type,
        )

    def postprocess_f(self):
        self.postprocess = Postprocessor(
            task="det", 
            algo=self.det_algorithm, 
            box_type=self.det_box_type,
        )

    def load_model(self, **args):
        if self.det_model_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            pretrained = False
            ckpt_load_path = get_ckpt_file(self.det_model_dir)
        
        self.model = build_model(
            self.model_name,
            pretrained=pretrained,
            pretrained_backbone=False,
            ckpt_load_path=ckpt_load_path,
            amp_level=self.det_amp_level,
        )
        return self.model

    def infer(self, img_or_path, **kwargs):
        self.model.set_train(mode=False)
        data = self.preprocess(img_or_path)
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)
        net_output = self.model(ms.Tensor(net_input))
        det_res = self.postprocess(net_output, data)
        det_res_final = validate_det_res(
            det_res,
            data["image_ori"].shape[:2], 
            min_poly_points=3, 
            min_area=3
        )
        return det_res_final
        

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass