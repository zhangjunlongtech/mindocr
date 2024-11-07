# import mindspore as ms
# import numpy as np
# from mindspore import Tensor, ops, mint
# import mindspore.mint.nn.functional as F
# ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)

# x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
# # output = ops.avg_pool2d(x, kernel_size=2, stride=1)
# # print(output.shape)

# out = F.avg_pool2d(x, kernel_size=2, stride=1, ceil_mode=True)
# # out = pool(x)
# print(out.shape)
import numpy as np
import mindspore
from mindspore import Tensor, Parameter, nn, ops
mindspore.set_context(mode=mindspore.GRAPH_MODE)
class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
                                name='weight')
        self.matmul = ops.MatMul()

    def construct(self, x):
        output = self.matmul(x, self.weight)
        return output

size, in_features, out_features = 16, 16, 10
#1) when the type of scale_sense is Cell:
net = Net(in_features, out_features)
loss_fn = nn.MSELoss()
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
net_with_loss = nn.WithLossCell(net, loss_fn)
input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
labels = Tensor(np.ones([out_features,]), mindspore.float32)
loss = net_with_loss(input, labels)
manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)
status = Tensor([0] * 8, mindspore.int32)
scaling_sens = train_network.scale_sense
scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
grads = train_network.grad(train_network.network, train_network.weights)(input, labels, scaling_sens_filled)
grads = train_network.grad_reducer(grads)
cond = train_network.get_overflow_status(status, grads)
overflow = train_network.process_loss_scale(cond)

#2) when the type of scale_sense is Tensor:
# net = Net(in_features, out_features)
# loss = nn.MSELoss()
# optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
# net_with_loss = nn.WithLossCell(net, loss)
# inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
# label = Tensor(np.zeros([size, out_features]).astype(np.float32))
# scaling_sens = Tensor([1024], dtype=mindspore.float32)
# train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scaling_sens)
# scaling_sens = Tensor([1], dtype=mindspore.float32)
# train_network.set_sense_scale(scaling_sens)
# output = train_network(inputs, label)

# # update scaling sens and train the network
# scaling_sens = Tensor([1], dtype=mindspore.float32)
# train_network.set_sense_scale(scaling_sens)
# output = train_network(inputs, label)
print("done")
