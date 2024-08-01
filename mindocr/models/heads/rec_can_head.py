import mindspore as ms
import math
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import unittest

"""
Counting Module
"""
class ChannelAtt(nn.Cell):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.SequentialCell([
            nn.Dense(channel, channel // reduction),
            nn.ReLU(),
            nn.Dense(channel // reduction, channel),
            nn.Sigmoid()
        ])

    def construct(self, x):
        b, c, _, _ = x.shape
        y = ops.reshape(self.avg_pool(x), (b, c))
        y = ops.reshape(self.fc(y), (b, c, 1, 1))
        return x * y


class CountingDecoder(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.trans_layer = nn.SequentialCell([
            nn.Conv2d(
                self.in_channel,
                512,
                kernel_size=kernel_size,
                pad_mode='pad',
                padding=kernel_size // 2,
                has_bias=False,
            ),
            nn.BatchNorm2d(512)
        ])

        self.channel_att = ChannelAtt(512, 16)

        self.pred_layer = nn.SequentialCell([
            nn.Conv2d(
                512,
                self.out_channel,
                kernel_size=1,
                has_bias=False,
            ),
            nn.Sigmoid()
        ])

    def construct(self, x, mask):
        b, _, h, w = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)

        if mask is not None:
            x = x * mask
        x = ops.reshape(x, (b, self.out_channel, -1))
        x1 = ops.sum(x, -1)

        return x1, ops.reshape(x, (b, self.out_channel, h, w))


"""
Attention Module
"""
class Attention(nn.Cell):
    def __init__(self, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.hidden = hidden_size
        self.attention_dim = attention_dim
        self.hidden_weight = nn.Dense(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(
            1,
            512,
            kernel_size=11,
            pad_mode='pad',
            padding=5,
            has_bias=False
        )
        self.attention_weight = nn.Dense(512, self.attention_dim, has_bias=False)
        self.alpha_convert = nn.Dense(self.attention_dim, 1)


    def construct(
            self, cnn_features, cnn_features_trans, hidden, alpha_sum, image_mask=None
    ):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0, 2, 3, 1))
        query_expanded = ops.unsqueeze(ops.unsqueeze(query, 1), 2)
        alpha_score = ops.tanh(
            query_expanded
            + coverage_alpha
            + cnn_features_trans.permute(0, 2, 3, 1)
        )
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = ops.exp(energy.squeeze(-1))
        if image_mask is not None:
            energy_exp = energy_exp * image_mask.squeeze(1)

        alpha = energy_exp / (
            ops.unsqueeze(ops.unsqueeze(ops.sum(ops.sum(energy_exp, -1), -1), 1), 2) + 1e-10
        )
        alpha_sum = ops.unsqueeze(alpha, 1) + alpha_sum
        context_vector = ops.sum(
            ops.sum((ops.unsqueeze(alpha, 1) * cnn_features), -1), -1
        )

        return context_vector, alpha, alpha_sum


"""
Attention Decoder
"""
class PositionEmbeddingSine(nn.Cell):
    def __init__(
            self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True when scale is provided")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, x, mask):
        y_embed = ops.cumsum(mask, 1, dtype=ms.float32)
        x_embed = ops.cumsum(mask, 2, dtype=ms.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = ops.arange(self.num_pos_feats, dtype=ms.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = ops.unsqueeze(x_embed, 3) / dim_t
        pos_y = ops.unsqueeze(y_embed, 3) / dim_t

        pos_x = ops.flatten(
            ops.stack(
                [ops.sin(pos_x[:, :, :, 0::2]), ops.cos(pos_x[:, :, :, 1::2])],
                axis=4,
            ),
            'C',
            start_dim=3,
        )
        pos_y = ops.flatten(
            ops.stack(
                [ops.sin(pos_y[:, :, :, 0::2]), ops.cos(pos_y[:, :, :, 1::2])],
                axis=4,
            ),
            'C',
            start_dim=3,
        )

        pos = ops.concat([pos_x, pos_y], axis=3)
        pos = ops.transpose(pos, (0, 3, 1, 2))
        return pos


class AttDecoder(nn.Cell):
    def __init__(
            self,
            ratio,
            is_train,
            input_size,
            hidden_size,
            encoder_out_channel,
            dropout,
            dropout_ratio,
            word_num,
            counting_decoder_out_channel,
            attention,
    ):
        super(AttDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_channel = encoder_out_channel
        self.attention_dim = attention["attention_dim"]
        self.dropout_prob = dropout
        self.ratio = ratio
        self.word_num = word_num
        self.counting_num = counting_decoder_out_channel
        self.is_train = is_train

        self.init_weight = nn.Dense(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_attention = Attention(self.hidden_size, self.attention_dim)

        self.encoder_feature_conv = nn.Conv2d(
            self.out_channel,
            self.attention_dim,
            kernel_size=attention["word_conv_kernel"],
            padding=attention["word_conv_kernel"] // 2,
        )

        self.word_state_weight = nn.Dense(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Dense(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Dense(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Dense(self.counting_num, self.hidden_size)
        self.word_convert = nn.Dense(self.hidden_size, self.word_num)

        if dropout:
            self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, cnn_features, labels, counting_preds, images_mask, is_train=True):
        if self.is_train:
            _, num_steps = labels.shape
        else:
            num_steps = 36

        batch_size, _, height, width = cnn_features.shape
        images_mask = images_mask[:, :, :: self.ratio, :: self.ratio]

        word_probs = ops.zeros((batch_size, num_steps, self.word_num))
        word_alpha_sum = ops.zeros((batch_size, 1, height, width))

        hidden = self.init_hidden(cnn_features, images_mask)
        counting_context_weighted = self.counting_context_weight(counting_preds)
        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])

        cnn_features_trans =cnn_features_trans + pos

        word = ops.ones([batch_size, 1], dtype=ms.int64)
        word = self.squeeze(axis=1)

        for i in range(num_steps):
            word_embedding = self.embedding(word)
            _, hidden = self.word_input_gru(word_embedding, hidden)
            word_context_vec, _, word_alpha_sum = self.word_attention(
                cnn_features,
                cnn_features_trans,
                hidden,
                word_alpha_sum,
                images_mask
            )

            current_state = self.word_state_weight(hidden)
            word_weight_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)

            if self.dropout_prob:
                word_out_state = self.dropout(
                    current_state
                    + word_weight_embedding
                    + word_context_weighted
                    + counting_context_weighted
                )
            else:
                word_out_state = (
                    current_state
                    + word_weight_embedding
                    + word_context_weighted
                    + counting_context_weighted
                )

            word_prob = self.word_convert(word_out_state)
            word_probs[:, i] = word_prob

            if self.is_train:
                word = labels[:, i]
            else:
                word = word_prob.argmax(1)
                word = ops.multiply(
                    word, labels[:, i]
                )

        return word_probs


    def init_hidden(self, features, feature_mask):
        average = ops.sum(
            ops.sum(features * feature_mask, dim=-1), dim=-1
        ) / ops.sum((ops.sum(feature_mask, dim=-1)), dim=-1)
        average = self.init_weight(average)
        return ops.tanh(average)


"""
Test Module
"""
def test_channelAtt():
    x = ms.Tensor(np.random.randn(32, 64, 32, 32), ms.float32)
    channel = ChannelAtt(64, 16)
    print(channel)
    output = channel(x)
    print(output.asnumpy())
    print(output.shape)
    print("done")


def test_countingDecoder():
    # 初始化CountingDecoder实例
    decoder = CountingDecoder(in_channel=3, out_channel=2, kernel_size=3)

    # 创建一个模拟输入张量和掩码张量
    x = ms.Tensor(np.random.rand(1, 3, 10, 10).astype(np.float32))
    mask = ms.Tensor(np.random.rand(1, 2, 10, 10).astype(np.float32))

    # 调用forward方法
    x1, x_reshaped = decoder(x, mask)
    print(x1.shape)

    # # 验证输出形状
    assert x1.shape == (1, 2), "The shape of x1 should be [1, 2]"
    assert x_reshaped.shape == (1, 2, 10, 10), "The shape of x_reshaped should be [1, 2, 10, 10]"

    # 验证输出内容（这里只是简单的非空验证，实际项目中可能需要更严格的验证）
    assert not np.all(x1.numpy() == 0), "The output x1 should not be all zeros"
    assert not np.all(x_reshaped.numpy() == 0), "The output x_reshaped should not be all zeros"
    print("done")

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.attModel = Attention(hidden_size=512, attention_dim=256)

    def test_initialization(self):
        self.assertEqual(self.attModel.hidden, 512)
        self.assertEqual(self.attModel.attention_dim, 256)
        self.assertTrue(isinstance(self.attModel.hidden_weight, nn.Dense))
        self.assertTrue(isinstance(self.attModel.attention_conv, nn.Conv2d))
        self.assertTrue(isinstance(self.attModel.alpha_convert, nn.Dense))

    def test_construct(self):
        cnn_featrues = ops.rand((1, 256, 14, 14))
        cnn_featrues_trans = ops.rand((1, 256, 14, 14))
        hidden = ops.rand((1, 512))
        alpha_sum = ops.zeros((1, 1, 14, 14))
        image_mask = ops.ones((1, 1, 14, 14))

        context_vextor, alpha, alpha_sum_out = self.attModel(cnn_featrues, cnn_featrues_trans, hidden, alpha_sum, image_mask)

        self.assertTrue(context_vextor.shape, (1, 512))
        self.assertTrue(alpha.shape, (1, 1, 14, 14))
        self.assertTrue(alpha_sum_out.shape, (1, 1, 14, 14))


if __name__ == '__main__':
    print("test start...")
    # test_channelAtt()
    # test_countingDecoder()
    unittest.main()
    print("test done!")