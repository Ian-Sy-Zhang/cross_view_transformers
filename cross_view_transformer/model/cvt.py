import torch.nn as nn


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False), # 二维卷积层，将输入特征映射的通道数从self.decoder.out_channels减少到dim_last。卷积核大小为3x3，填充为1，没有偏置项。
            nn.BatchNorm2d(dim_last), # 归一化层，用于调整上一层的输出特征映射的均值和方差，以提高模型的稳定性和收敛速度。
            nn.ReLU(inplace=True), # ReLU激活函数层，通过将小于零的值设为零来引入非线性。
            nn.Conv2d(dim_last, dim_max, 1)) # 二维卷积层，它将输入特征映射的通道数从dim_last增加到dim_max。卷积核大小为1x1，没有填充，没有偏置项。

    def forward(self, batch):
        x = self.encoder(batch) # 首先通过编码器模型self.encoder对输入进行编码
        y = self.decoder(x) # 通过解码器模型self.decoder对编码结果进行解码
        z = self.to_logits(y) # 最后通过self.to_logits模块将解码结果转换为输出（BEV产出）

        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
