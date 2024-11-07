import mindspore as ms   
import numpy as np  
  
# 加载检查点文件  
checkpoint_path = '/home/ma-user/work/mindocr0930/tests/ut/can_params_from_paddle.ckpt'  
param_dict = ms.load_checkpoint(checkpoint_path)  
  
# 遍历参数字典并打印参数名称和值  
for param_name, param_value in param_dict.items():  
    # 将 Tensor 转换为 NumPy 数组（可选）  
    param_value_numpy = param_value.asnumpy()  
      
    # 打印参数名称和形状（或者你可以打印整个数组，但这可能会很冗长）  
    print(f"Parameter Name: {param_name}")  
    print(f"Parameter Shape: {param_value_numpy.shape}")  
    print(f"Parameter Values (first few elements): {param_value_numpy[:5]}")  # 打印前几个元素作为示例  
    print("-" * 40)  # 分隔线 