import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import math
import numpy as np
from mindocr import build_postprocess


if __name__=="__main__":
    config = dict(name="CANLabelDecode", character_dict_path="/home/ma-user/work/mindocr0930/mindocr/utils/dict/latex_symbol_dict.txt")
    data_decoder = build_postprocess(config)
    print("done")


