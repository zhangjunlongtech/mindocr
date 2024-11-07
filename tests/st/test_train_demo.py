import subprocess
  
# 定义要执行的命令及其参数  
# 注意：在列表中分隔命令和它的参数  
command = [  
    'python',  
    '/home/ma-user/work/mindocr0919/tests/st/test_can_train.py',  
    '--config',  
    '/home/ma-user/work/mindocr0919/configs/rec/can/can_d28.yaml'  
]  
  
# 使用subprocess.run()执行命令  
# capture_output=True 捕获输出（标准输出和标准错误输出）  
# text=True 将输出作为文本处理（Python 3.7+）  
result = subprocess.run(command, capture_output=True, text=True) 
print(result.stdout)  