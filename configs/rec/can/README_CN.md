[English]() | 中文

# 手写数学公式识别算法-CAN

- [手写数学公式识别算法-CAN](#手写数学公式识别算法-can)
  - [1. 算法简介](#1-算法简介)
  - [2. 环境配置](#2-环境配置)
  - [3. 模型推理、评估、训练](#3-模型推理评估训练)
    - [3.1 推理](#31-推理)
    - [3.2 评估](#32-评估)
    - [3.3 训练](#33-训练)
  - [4. FAQ](#4-faq)
  - [引用](#引用)
  - [参考文献](#参考文献)



## 1. 算法简介
<!--- Guideline: Introduce the model and architectures. Cite if you use/adopt paper explanation from others. -->
> [CAN: When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/pdf/2207.11463.pdf)

CAN是具有一个弱监督计数模块的注意力机制编码器-解码器手写数学公式识别算法。本文作者通过对现有的大部分手写数学公式识别算法研究，发现其基本采用基于注意力机制的编码器-解码器结构。该结构可使模型在识别每一个符号时，注意到图像中该符号对应的位置区域，在识别常规文本时，注意力的移动规律比较单一（通常为从左至右或从右至左），该机制在此场景下可靠性较高。然而在识别数学公式时，注意力在图像中的移动具有更多的可能性。因此，模型在解码较复杂的数学公式时，容易出现注意力不准确的现象，导致重复识别某符号或者是漏识别某符号。

针对于此，作者设计了一个弱监督计数模块，该模块可以在没有符号级位置注释的情况下预测每个符号类的数量，然后将其插入到典型的基于注意的HMER编解码器模型中。这种做法主要基于以下两方面的考虑：1、符号计数可以隐式地提供符号位置信息，这种位置信息可以使得注意力更加准确。2、符号计数结果可以作为额外的全局信息来提升公式识别的准确率。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/miss_word.png" width=640 />
</p>
<p align="center">
  <em> 图1. 手写数学公式识别算法对比 [<a href="#参考文献">1</a>] </em>
</p>

CAN模型由主干特征提取网络、多尺度计数模块（MSCM）和结合计数的注意力解码器（CCAD）构成。主干特征提取通过采用DenseNet得到特征图，并将特征图输入MSCM，得到一个计数向量（Counting Vector），该计数向量的维度为1*C，C即公式词表大小，然后把这个计数向量和特征图一起输入到CCAD中，最终输出公式的latex。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/total_process.png" width=640 />
</p>
<p align="center">
  <em> 图2. 整体模型结构 [<a href="#参考文献">1</a>] </em>
</p>

多尺度计数模MSCM块旨在预测每个符号类别的数量，其由多尺度特征提取、通道注意力和池化算子组成。由于书写习惯的不同，公式图像通常包含各种大小的符号。单一卷积核大小无法有效处理尺度变化。为此，首先利用了两个并行卷积分支通过使用不同的内核大小（设置为 3×3 和 5×5）来提取多尺度特征。在卷积层之后，采用通道注意力来进一步增强特征信息。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/MSCM.png" width=640 />
</p>
<p align="center">
  <em> 图3. MSCM多尺度计数模块 [<a href="#参考文献">1</a>] </em>
</p>

结合计数的注意力解码器：为了加强模型对于空间位置的感知，使用位置编码表征特征图中不同空间位置。另外，不同于之前大部分公式识别方法只使用局部特征进行符号预测的做法，在进行符号类别预测时引入符号计数结果作为额外的全局信息来提升识别准确率。

<p align="center">
  <img src="https://temp-data.obs.cn-central-221.ovaijisuan.com/mindocr_material/CCAD.png" width=640 />
</p>
<p align="center">
  <em> 图4. 结合计数的注意力解码器CCAD [<a href="#参考文献">1</a>] </em>
</p>

<a name="model"></a>
`CAN`使用CROHME手写公式数据集进行训练，在对应测试集上的精度如下：

|模型    |骨干网络|配置文件|ExpRate|下载链接|
| ----- | ----- | ----- | ----- | ----- |
|CAN|DenseNet|[rec_d28_can.yml](../../configs/rec/rec_d28_can.yml)|51.72%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|

<a name="2"></a>
## 2. 环境配置
请先参考配置MindOCR运行环境，其中MindSpore需要支持2.4.0


<a name="3"></a>
## 3. 模型推理、评估、训练

<a name="3-1"></a>
### 3.1 推理

首先将训练得到best模型，转换成inference model。这里以训练完成的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar) )，可以使用如下命令进行转换：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/rec/rec_d28_can.yml -o Global.pretrained_model=./rec_d28_can_train/best_accuracy.pdparams Global.save_inference_dir=./inference/rec_d28_can/ Architecture.Head.attdecoder.is_train=False

# 目前的静态图模型默认的输出长度最大为36，如果您需要预测更长的序列，请在导出模型时指定其输出序列为合适的值，例如 Architecture.Head.max_text_length=72
```
**注意：**
- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。

转换成功后，在目录下有三个文件：
```
/inference/rec_d28_can/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

执行如下命令进行模型推理：

```shell
python3 tools/infer/predict_rec.py --image_dir="./doc/datasets/crohme_demo/hme_00.jpg" --rec_algorithm="CAN" --rec_batch_num=1 --rec_model_dir="./inference/rec_d28_can/" --rec_char_dict_path="./ppocr/utils/dict/latex_symbol_dict.txt"

# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/datasets/crohme_demo/'。

# 如果您需要在白底黑字的图片上进行预测，请设置 --rec_image_inverse=False
```

![测试图片样例](../datasets/crohme_demo/hme_00.jpg)

执行命令后，上面图像的预测结果（识别的文本）会打印到屏幕上，示例如下：
```shell
Predicts of ./doc/imgs_hme/hme_00.jpg:['x _ { k } x x _ { k } + y _ { k } y x _ { k }', []]
```


**注意**：

- 需要注意预测图像为**黑底白字**，即手写公式部分为白色，背景为黑色的图片。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中CAN的预处理为您的预处理方法。

<a name="3-2"></a>
### 3.2 评估

可下载已训练完成的[模型文件](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)，使用如下命令进行评估：

```shell
python tools/eval.py --config configs/rec/can/can_d28.yaml
```

<a name="3-1"></a>
### 3.3 训练

请参考文本识别训练教程。MindOCR对代码进行了模块化，训练`CAN`识别模型时需要**更换配置文件**为`CAN`的[配置文件](../../configs/rec/xxx.yml)。


具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
python tools/train.py --config configs/rec/can/can_d28.yaml

**注意：**
- 我们提供的数据集，即[`CROHME数据集`](https://paddleocr.bj.bcebos.com/dataset/CROHME.tar)将手写公式存储为黑底白字的格式，若您自行准备的数据集与之相反，训练前请统一处理数据集。
```


<a name="5"></a>
## 4. FAQ

1. CROHME数据集来自于[CAN源repo](https://github.com/LBH1024/CAN) 。

## 引用

```bibtex
@misc{https://doi.org/10.48550/arxiv.2207.11463,
  doi = {10.48550/ARXIV.2207.11463},
  url = {https://arxiv.org/abs/2207.11463},
  author = {Li, Bohan and Yuan, Ye and Liang, Dingkang and Liu, Xiao and Ji, Zhilong and Bai, Jinfeng and Liu, Wenyu and Bai, Xiang},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


## 参考文献
<!--- Guideline: Citation format GB/T 7714 is suggested. -->
[1] Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne Zhang. RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition. arXiv:2007.07542, ECCV'2020
