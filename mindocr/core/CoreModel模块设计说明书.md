
| _***日期***_ | _***修订版本***_ | _***修改章节***_ | _***修改描述***_ | _***作者***_                                            |
| ---------- | ------------ | ------------ | ------------ | ----------------------------------------------------- |
| 2024-08-19 | 1.0          |              |        | zhangjunlong
待更新补充
## 前言
---
MindSpore OCR（以下简称MindOCR）套件是基于MindSpore AI框架开发的OCR开源工具包，集成文本检测、文本识别等业界主流算法模型，并提供端到端的应用工具，帮助用户快速开发OCR领域模型，满足用户通用OCR场景与复杂文档分析场景的生产落地需求。

围绕MindOCR的产品定位，本人的工作范围包括但不限于：
1. 负责MindOCR套件自有模型丰富度提升：实现1~2个关键模型，打通模型迁移流程，降低模型迁移门槛，牵引社区开源贡献，最终实现模型丰富度的显著提高；
2. 负责通用OCR场景使用与开发的易用性提升：提升套件使用的便捷性，降低套件开发的复杂度，使MindOCR套件更好服务于生产落地需求；
3. 负责MindOCR动态图模式的支持；
4. 负责MindOCR语言分析系统的支持；

经实践发现，MindOCR在910B与MindSpore2.3.0的环境下已能够较好地支持动态图模式；语言分析系统在实际生产落地的需求中优先级较低，故第3、4项任务暂时不在主要工作中呈现。

由此，本人重点工作项如下图黄色模块所示：

<p align="center">
  <img src="https://gitee.com/junlong1/testBag/raw/master/mindocr_material/mindocr_todepic.jpg
  " width=640 />
</p>
<p align="center">
  <em> 图1. 重点工作示意图 </em>
</p>

其中，整图为当前MindOCR工作全景图，绿色模块为当前MindOCR套件已支持的部分，黄色模块与灰色模块为待重点开展的工作，其中黄色模块为本人重点工作。

本文档主要围绕图1中的coremodel模块展开。coremodel模块属于通用OCR系统基础库下的子模块，旨在提升MindOCR开发与使用的易用性，提升套件使用的便捷性，降低套件开发的复杂度，使MindOCR套件更好服务于生产落地需求。本文将从背景与目的、上下文定义、总体结构、模块设计几个层面进行描述。

## 一、简介
-------
### 1.1 背景

1、调用入口多，使用复杂度高

当前MindOCR支持Det、Rec、KIE、Layout、Table等多种模型的推理、训练与评估任务，对于各类任务推理分别需调用不同的推理脚本（如`predict_det`, `predict_rec`, `predict_ser`, `predict_system`等），而对于训练与评估，同样需要分别调用对应的训练与评估脚本。随着MindOCR模型库和新功能的支持（如智能文档分析系统），大量不同的调用入口将给开发者的使用带来较高的复杂度，且开发者在安装mindocr包后无法直接调用API进行训练、评估、推理，这将导致用户开发的快速性与便捷性不足，亟需得到改善。

我们希望用户可以基于统一的类对象去实现各类任务（如模型加载、推理、训练与评估），用户在安装mindocr后可直接调用几行API即可进行推理、训练、评估，降低用户使用复杂度，提升用户开发的快速性与便捷性。

2、用户开发自由度不足

当前MindOCR在模型训练和推理时，为了调用tools目录下的推理训练脚本通常需要将项目全量clone至本地，而无法通过导入mindocr包后直接调用。当用户需要开发自己的模型或更换部分模块/功能时，也需要将整个mindocr项目全量clone至本地进行模型迁移式开发。

我们希望通过设计一个抽象类，开发者在pip安装导包后，仅需通过提供的API接口写入自己的私有模型或组件即可完成开发，以此提高用户在开发过程中的自由度。

3、缺少输入图像前置处理模块，灵活性不足

MindOCR当前缺少对输入图像的预处理模块，如格式转化、大图切片、透明填充、图像二值化、颜色反转、角度调整、批量推理等。

我们希望用户在输入图像时可根据自身需要对输入进行处理，以提高用户使用精度。


### 1.2 目的

当前MindOCR存在模型推理、训练、评估时入口复杂，且无法通过API的形式调用等问题。且开发者无法以实现接口的形式实现自己私有的模型或特定的功能。

针对这些问题，本文档进行了MindOCR CoreModel模块的设计，通过统一的CoreModel类抽象出模型加载、数据前处理、数据后处理、推理、训练、评估等方法。开发者可以基于已有的API对mindocr库中已有的模型进行训练、推理，也可基于抽象类接口实现自己的模型训练与推理。

除此之外，本设计文档还提供了图片预处理工具。

1、当前推理调用方式如下：
```
python tools/infer/text/predict_system.py --image_dir {path_to_img or dir_to_imgs}  --det_algorithm DB++  --rec_algorithm CRNN
```

支持CoreModel模块后，可通过如下方式运行MindOCR图像识别功能。其中RecModel为CoreModel子类，用于执行文字识别相关任务。
```python
from mindocr import RecModel
model = RecModel(rec='CRNN', **args) 
result = model.infer(img_path, **args)
```

2、当前训练调用方式如下：
```
python tools/train.py --config path/to/model_train_config.yaml
```

支持CoreModel模块后，训练调用方式如下。其中，DetModel为CoreModel子类，用于实现文字检测任务。
```python
from mindocr import DetModel
model = DetModel(det='DB', **args)
model.train(data_path, ckpt_path, **args)
```

用户亦可通过接口定义自己的前处理、后处理以及模型等模块，通过接口训练自有模型：
```python
from mindocr import CoreModel
class UserModel(CoreModel)：
    def load_model(**args):
      pass

    def preprocess(**args):
      pass

    def postprocess(**args):
      pass

model = UserModel(**args)
model.train(data_path, ckpt_path, **args)
```

3、当前评估流程如下：
```
python tools/eval.py --config path/to/config.yaml
--opt dataset_root  ckpt_load_path
```
支持CoreModel模块后，评估流程可通过如下形式实现：
```python
from mindocr import RecModel
model = RecModel(alg='CRNN', **args)
model.eval(data_path, ckpt_path, **args)
```

由此，通过MindOCR CoreModel模块下的CoreModel抽象类，将模型各行为进行统一继承，实现模型的训推一体化。

### 1.4 范围
#### 1.4.1 软件名称
MindOCR CoreModel模块
#### 1.4.2 软件功能
1. 用户通过统一API接口实现模型的推理、训练、评估等；
2. 用户通过抽象类的API接口实现自有模块的开发；
2. 支持OCR输入图像预处理功能

## 二、第零层设计描述
---
### 2.1 软件系统上下文定义
安装的环境要求：同 MindSpore OCR 安装的环境要求相同。

如图1所示，原有项目进行mindocr目录下模型推理时，需要调用tools目录下的工具脚本，如在进行图像检测识别时，需通过命令行执行tools/infer/text目录下的predict_sysytem脚本。
新增的MindOCR CoreModel模块在mindocr目录下，其中包含了CoreModel等抽象类及其相关的组件。新增模块对原有tools功能无影响。
```mermaid
graph TD
    A[MindSpore OCR 仓库]
    B --> E[MindOCR CoreModel module]
    A --> C[tools 目录]
    A --> G[mindviewer 目录]
    A --> B[mindocr 目录]
```
<p align="center">
  <em> 图1. MindOCR目录结构 </em>
</p>

### 2.2 设计约束
#### 2.2.1 遵循标准
遵循Apache 2.0
#### 2.2.2	硬件限制
Ascend：910A、910B
GPU：支持cuda10.1以上
CPU
#### 2.2.3 技术限制
GPU、CPU暂不支持MindSpore 2.3
#### 2.2.4 系统配置
openEuler: >= 20.3
CentOS：>= 7.6
Ubuntu：>= 18.04
Windows 10
MacOS：10.15/11.3


# 三、第一层设计描述
---
## 3.1 总体结构

MindOCR CoreModel模块整体架构如下图所示
```mermaid
graph LR
    subgraph MindOCR CoreModel Module
    A[CoreModel]
    B[DetModel]
    C[RecModel]
    D[KIEModel]
    E[CLSModel]
    F[TabModel]
    G[LayModel]
    end
    B --> A
    C --> A
    D --> A
    E --> A
    F --> A
    G --> A
```
<p align="center">
  <em> 图2. MindOCR CoreModel模块总体结构 </em>
</p>

## 3.2 分解描述
+ CoreModel：
模型统一抽象类，抽象出各类模型模型加载、数据前后处理、推理、训练、评估的通用方法，开发者可基于该抽象类实现自己的私有模块；
+ DetModel：
检测模型统一调用入口，用于文本检测相关任务;
+ RecModel：
识别模型统一调用入口，用于文本识别相关任务;
+ KIEModel：
关键信息提取模型统一调用入口，用于文本关键信息提取相关任务;
+ CLSModel：
文本方向分类模型统一调用入口，用于文本方向分类相关任务;
+ TabModel：
表格识别模型统一调用入口，用于表格识别相关任务;
+ LayModel：
版面分析模型统一调用入口，用于版面分析相关任务;


## 3.3 运行设计

1、原有项目运行流程

当前（原有）运行流程如下图所示（以文本检测识别任务为例）。predict_system为文字检测识别调用工具，用户通过调用该入口工具进入TextOCR流程，该流程首先通过predict_det（检测入口）运行mindocr中的文字检测模型，再将检测结果传入predict_rec（识别入口）中，识别模块调用mindocr中的文字识别模型最终输出结果。
对于不同的任务，用户需要调用该任务特定的执行工具，缺少统一运行入口。

```mermaid
graph LR
    A["Image"] --> B["predict_system"]
    C["config"] --> B
    B --> D
    I --> J["Output"]
    E <--> K["mindocr"]
    H <--> L["mindocr"]
  
  subgraph TextOCR
    subgraph Detection
      D["PreProcessor"] --> E["TextDetector"]
      E --> F["PostProcessor"]
    end

    subgraph Recognition
      F --> G["PreProcessor"]
      G --> H["TextRecognizer"]
      H --> I["PostProcessor"]
    end
  end

```
<p align="center">
  <em> 图3. 原有文字识别运行流程 </em>
</p>

2、新增MindOCR CoreModel模块后运行流程

通过OCRCore模块的设计，将原有分散的推理、训练、评估流程进行了统一，开发者在导入包后可直接通过调用接口进行模型调用，也可利用接口自行开发模型。运行流程图如下：

（运行流程图待补充）


# 四、第二层设计描述
---
## 4.1 CoreModel模块设计

#### 4.1.1 总体结构
待补充


#### 4.1.2 模块分解描述
待补充
 

#### 4.1.3 数据实体描述

```mermaid

classDiagram
  class CoreModel {
    +mode: int
    +args: dict
    +preprocessor: PreProcessor
    +postprocessor: PostProcessor

    +__init__(): void
    +load_model(): void
    +preprocess(): void
    +postprocess(): void
    +infer(): void
    +train(): void
    +eval(): void
    +jsonify(): void
  }

  class TextModel{
    +det_alg: str
    +rec_alg: str
    +args: dict
    +__init__(): void
    +infer():void
  }

  class PostProcessor{
    +task: str
    +data: str
    +args: dict
    +__init__(): void
    +__call__(): void
  }

  class PreProcessor{
    +task: str
    +data: str
    +args: dict
    +__init__(): void
    +__call__(): void
  }

  CoreModel <|-- DetModel
  CoreModel <|-- RecModel
  CoreModel <|-- CLSModel
  CoreModel <|-- KIEModel
  CoreModel <|-- TabModel
  CoreModel <|-- LayModel
  TextModel *-- DetModel
  TextModel *-- RecModel
  CoreModel *-- PreProcessor
  CoreModel *-- PostProcessor
```

<p align="center">
  <em> 图xxx </em>
</p>

CoreModel抽象类有如下公共变量：

- mode：运行模式，0为graph模式，1为pynative模式
- args：配置参数，用户自定义或系统默认值
- preprocessor：前处理过程
- postprocessor：后处理过程
- __init__()：类的初始化方法
- load_model()：模型加载，用户可自定义
- preprocess()：模型前处理过程，用户可自定义
- infer()：模型推理过程，用户可自定义
- train()：模型训练方法，用户可自定义
- eval(): 模型评估方法，用户可自定义
- jsonify(): 将输出转为json格式


#### 4.1.4 依赖性描述

外部第三方依赖库：opencv、numpy

#### 4.1.5 接口使用描述

待补充

## 4.2 ImgFormatter模块设计

#### 4.2.1 总体结构
待补充


#### 4.2.2 模块分解描述
待补充


#### 4.2.3 数据实体描述

```mermaid
classDiagram 
  class ImgFormatter{
    +cls: bool
    +bin: bool
    +inv: bool
    +alpha_color: bool
    +slice: bool
    +__init__(): void
    +image_preprocess: void
    +conv_img(): void
    +binarize_img(): void
    +cls_img(): void
    +slice_img(): void
  }

```

<p align="center">
  <em> 图xxx </em>
</p>

ImgFormatter类有如下公共变量：

- `cls`: 是否使用角度分类器。默认为True
- `bin`: 将图像二值化为黑白。默认为False
- `inv`: 反转图像颜色。默认为False。
- `alpha_color`: 设置透明部件更换的RGB颜色元组。默认为纯白色
- `slice`: 对大图像使用滑动窗口推理。
- `__init__()`: 初始化ImgFormatter类成员变量
- `__call__()`: 图像前处理操作pipeline
- `conv_img()`: 图像格式转换，可支持gif、pdf文件转换
- `binarize_img()`: 图像二值化方法
- `cls_img()`: 角度分类器方法
- `slice_img()`: 图像切片滑动窗口处理方法


#### 4.2.4 依赖性描述

外部第三方依赖库：opencv、numpy

#### 4.2.5 接口使用描述
待补充
