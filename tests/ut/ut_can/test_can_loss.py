import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import math
import numpy as np
from mindocr import build_loss

ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
# 假设 CANLoss 类已经定义如上

# 模拟一些数据
# 假设 batch_size = 4, seq_length = 10, vocab_size = 111
batch_size = 1
seq_length = 10
vocab_size = 111

# 模拟 preds
word_probs = ops.randn((batch_size, seq_length, vocab_size))  # 假设是 logits
counting_preds = ops.randn((batch_size, vocab_size))
counting_preds1 = ops.randn((batch_size, vocab_size))
counting_preds2 = ops.rand((batch_size, vocab_size))

# 模拟 labels 和 labels_mask
labels = ops.randint(low=0, high=vocab_size, size=(batch_size, seq_length), dtype=ms.int32)
labels_mask = ops.randn((batch_size, seq_length)) > 0.5  # 假设有一半的 token 是有效的
labels_mask = labels_mask.astype('float32')

# 构造 batch 数据
batch = [labels, labels_mask]  # 忽略前两个 None，因为在这个测试中我们不需要它们

# 创建 CANLoss 实例
loss_fn = build_loss("CANLoss")

preds=dict()
preds["word_probs"]=word_probs
preds["counting_preds"]=counting_preds
preds["counting_preds1"]=counting_preds1
preds["counting_preds2"]=counting_preds2


# 调用 forward 方法
# loss_dict = loss_fn(preds=[word_probs, counting_preds, counting_preds1, counting_preds2], batch=batch)
loss_dict = loss_fn(preds, *batch)

# 输出损失
print(loss_dict)