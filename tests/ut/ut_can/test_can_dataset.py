import logging
import os
import random
from typing import List, Union

import numpy as np
from scipy.io import loadmat

import sys
import mindocr
import mindspore as ms
from mindocr.data.transforms.transforms_factory import create_transforms, run_transforms
from mindocr.data import build_dataset


sys.path.append(".")
ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

if __name__ == "__main__":
    # model parameter setting
    # transform_pipeline = [
    #     {"DecodeImage": {
    #         "img_mode": "BGR",
    #         "channel_first": False,
    #         },
    #     },

    #     {"NormalizeImage": {
    #         "mean": [0,0,0],
    #         "std": [1,1,1],
    #         "is_hwc": True,
    #         },
    #     },

    #     {"GrayImageChannelFormat": {
    #         "inverse": True
    #         },
    #     },

    #     {"CANLabelEncode":{
    #         "lower": False,
    #         "character_dict_path": "/home/ma-user/work/prtest/mindocr/mindocr/utils/dict/latex_symbol_dict.txt"
    #         }
    #     },

    #     {"KeepKeys": {
    #         "keep_keys": ["image", "label"]
    #         }
    #     },
    # ]

    # data = {
    #     "img_path": "/home/ma-user/work/prtest/mindocr/rec_test2.png",
    #     "label": "\sqrt { a } = 2 ^ { - n } \sqrt { 4 ^ { n } a }"
    # }

    # global_config = dict(is_train=True, use_minddata=False)
    # transforms = create_transforms(transform_pipeline, global_config)
    # data = run_transforms(data, transforms=transforms)

    # print(data[0].shape)
    # print(data[1])
    # print("done")
    
    data_config = {
        "type": "RecDataset",
        "dataset_root": "/home/nginx/work/zhangjunlong/mindocr0930/myds",
        "data_dir": "/home/nginx/work/zhangjunlong/mindocr0930/myds/training",
        "label_file": "/home/nginx/work/zhangjunlong/mindocr0930/myds/gt_training.txt",
        "sample_ratio": 1.0,
        "shuffle": False,
        "transform_pipeline": [
            {"DecodeImage": {
                "img_mode": "BGR",
                "channel_first": False,
                },
            },

            {"CANImageNormalize": {
                "mean": [0,0,0],
                "std": [1,1,1],
                "order": 'hwc',
                },
            },

            {"GrayImageChannelFormat": {
                "inverse": True
                },
            },

            {"CANLabelEncode":{
                "is_train": True,
                "lower": False,
                "character_dict_path": "/home/nginx/work/zhangjunlong/mindocr0930/mindocr/utils/dict/latex_symbol_dict.txt"
                }
            },
            
        ],
        "output_columns": ["image", "label"],
        "net_input_column_index": [0,1,2],
        "label_column_index": [2,3],
    }

    loader_config = {
        "shuffle": False,
        "batch_size": 1,
        "drop_remainder": False,
        "num_workers": 1,
        "collate_fn": "can_collator",
        "output_columns": ["images", "image_masks", "labels", "label_masks"],
    }
    # loader_config = dict(shuffle=False, batch_size=4, drop_remainder=False, num_workers=1, collate_fn="can_collator")
    data_loader = build_dataset(data_config, loader_config, num_shards=1, shard_id=0, is_train=True)
    print(f"size={data_loader.get_dataset_size()}")
    print(f"col_name={data_loader.get_col_names()}")

    # for batch, data in enumerate(data_loader.create_dict_iterator()):
        
    
    for data in data_loader.create_dict_iterator():
        print("----------")
        print(type(data))
        print(type(data["images"]))
        print(data["images"].shape)
        print(data["image_masks"].shape)
        print(data["labels"].shape)
        print(data["label_masks"].shape)
        # print(data["labels"].dtype)
        # print(data["images"])
        # print(data["labels"])
        # print(data["image_masks"])
        # print(len(data["images"]))
        # print(len(data["image_masks"]))
        # print(len(data["labels"]))
        # print(len(data["label_masks"]))
        # print("data get 2")

    # data_eval_config = {
    #     "type": "RecDataset",
    #     "dataset_root": "/home/nginx/work/zhangjunlong/mindocr0930/myds",
    #     "data_dir": "/home/nginx/work/zhangjunlong/mindocr0930/myds/training",
    #     "label_file": "/home/nginx/work/zhangjunlong/mindocr0930/myds/gt_training.txt",
    #     "sample_ratio": 1.0,
    #     "shuffle": False,
    #     "transform_pipeline": [
    #         {"DecodeImage": {
    #             "img_mode": "BGR",
    #             "channel_first": False,
    #             },
    #         },

    #         {"NormalizeImage": {
    #             "mean": [0,0,0],
    #             "std": [1,1,1],
    #             "order": 'hwc',
    #             },
    #         },

    #         {"GrayImageChannelFormat": {
    #             "inverse": True
    #             },
    #         },

    #         {"CANLabelEncode":{
    #             "is_train": False
    #             },
    #         },
            
    #     ],
    #     "output_columns": ["image", "image_mask","ones_label_fill","label"],
    #     "net_input_column_index": ["image","image_mask","ones_label_fill"],
    #     "label_column_index": ["label"],
    # }

    # eval_loader_config = {
    #     "shuffle": False,
    #     "batch_size": 1,
    #     "drop_remainder": False,
    #     "num_workers": 1,
    # }

    # data_eval_loader = build_dataset(data_eval_config, eval_loader_config, num_shards=1, shard_id=0, is_train=False)
    # print(f"size={data_eval_loader.get_dataset_size()}")
    # print(f"col_name={data_eval_loader.get_col_names()}")
    # for data in data_eval_loader.create_dict_iterator():
    #     print(data["image"].shape)
    #     print(data["image_mask"].shape)
    #     print(data["ones_label_fill"].shape)
    #     print(data["label"])

    # def select_inputs_by_indices(inputs, indices):
    #     new_inputs = list()
    #     for x in indices:
    #         new_inputs.append(inputs[x])
    #     return new_inputs



    # for data in data_loader.create_tuple_iterator():
    #     for i in data:
    #         print(i.shape)

    # for batch, data in enumerate(data_loader.create_tuple_iterator()):
    #     print(type(data))
        # print(f"data shape:{data.shape}")
        # print(f"label shape:{label.shape}")
        # print(f"label:{label}")


    # print(list(data_loader))


    # print("zbc")

    # for dddt in data_loader.create_dict_iterator():
    #     print(dddt["image"])

    # for i in data_loader.create_tuple_iterator():
    #     print(i)
    #     print(f"data shape:{data.shape}")
    #     print(f"label shape:{label.shape}")
    #     print(f"label:{label}")
        # print(label_masks)

    # for batch, (data,label) in enumerate(data_loader.create_tuple_iterator()):
    #     print(batch)
    #     print(f"data shape:{data.shape}")
    #     print(f"label shape:{label.shape}")
    #     print(f"label:{label}")

    # for batch, data in enumerate(data_loader.create_tuple_iterator()):
    #     print(type(data))
        # print(f"data shape:{data.shape}")
        # print(f"label shape:{label.shape}")
        # print(f"label:{label}")
    
    # iterator = data_loader.create_tuple_iterator(num_epochs=2)
    # for epoch in range(2):
    #     for item in iterator:
    #         print(type(item))
    #     print("---")
        
    # next(data_loader.create_tuple_iterator())
    # data_loader.create_tuple_iterator()
    print("done")

