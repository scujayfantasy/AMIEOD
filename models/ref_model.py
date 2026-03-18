import torch
import torch.nn as nn
import torch.nn.functional as F

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    """传统问题：CNN和RNN因局部感受野或顺序计算的限制，难以建模远距离关系（如句子中相隔很远的词、图像中分离的物体）。"""
    """自注意力优势：直接计算任意两个位置的关系权重，无论距离多远（如Transformer中一句开头和结尾的词可直接交互）。"""
    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian'):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.mode = mode

        if self.inter_channels is None:
            self.inter_channels = in_channels // 32
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if mode in ['embedded_gaussian', 'dot_product']:
            self.theta = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.inter_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
            self.phi = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.inter_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        if mode == 'embedded_gaussian':
            self.softmax = nn.Softmax(dim=-1)
        elif mode == 'dot_product':
            pass

        self.W = nn.Conv2d(in_channels=self.inter_channels,
                           out_channels=self.in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode in ['embedded_gaussian', 'dot_product']:
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            if self.mode == 'embedded_gaussian':
                f = self.softmax(f)
            elif self.mode == 'dot_product':
                f = f / f.size(-1)

            y = torch.matmul(f, g_x)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class EFC_Optimized(nn.Module):
    def __init__(self, c1, c2, group_num=16):
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=1)
        self.bn = nn.BatchNorm2d(c2)
        self.sigmoid = nn.Sigmoid()

        # 分组归一化
        self.group_num = group_num
        self.gn = nn.GroupNorm(group_num, c2)

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(c2, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c2, 1, 1))

        # 门控生成器
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        # 动态卷积分支
        self.dwconv = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2)
        self.conv3 = nn.Conv2d(c2, c2, kernel_size=1)

        # 分组交互卷积
        self.interacts = nn.ModuleList([
            nn.Conv2d(c2 // 4, c2 // 4, kernel_size=1) for _ in range(4)
        ])

        # 全局特征压缩
        self.global_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, x2 = x

        # 分支特征提取
        global_conv1 = self.conv1(x1)
        weight_1 = self.sigmoid(self.bn(global_conv1))

        global_conv2 = self.conv2(x2)
        weight_2 = self.sigmoid(self.bn(global_conv2))

        # 全局特征融合
        X_GOBAL = global_conv1 + global_conv2

        # 分组注意力增强
        X_split = torch.chunk(X_GOBAL, 4, dim=1)
        out = []
        for i in range(4):
            x_group = self.interacts[i](X_split[i])
            N, C, H, W = x_group.shape
            x_attn = F.softmax(x_group.view(N, C, -1), dim=-1).view(N, C, H, W)
            out.append(x_group * x_attn)
        out = torch.cat(out, dim=1)

        # 归一化调整
        out = self.gn(out) * self.gamma + self.beta

        # 动态门控融合
        reweights = self.global_compress(X_GOBAL)
        mask_up = reweights >= weight_1
        mask_low = reweights < weight_1

        x_low = self.conv3(self.dwconv(X_GOBAL * mask_low)) * self.gate_genator(X_GOBAL)
        x_up = self.conv4(X_GOBAL * mask_up)

        return out + x_low + x_up


import torch
import torch.nn as nn
import torch.nn.functional as F
# 
# 
# class Enhance(nn.Module):
#     def __init__(self):
#         super(Enhance, self).__init__()
#
#         # 编码器（下采样）
#         self.enc_conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.enc_conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
#         self.enc_conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )
#
#         # 解码器（上采样 + 跳跃连接）
#         self.dec_up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec_conv1 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 输入通道=64(上采样)+64(跳跃)
#             nn.ReLU()
#         )
#         self.dec_up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.dec_conv2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输入通道=32(上采样)+32(跳跃)
#             nn.ReLU()
#         )
#         self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
#
#         # 可学习光照曲线（同前）
#         self.curve_params = nn.Parameter(torch.ones(3) * 0.5)
#
#         # self.fuse_net = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
#         #                                 nn.BatchNorm2d(32),
#         #                                 nn.ReLU(),
#         #                                 nn.Conv2d(32, 3, 3, 1, 1, groups=1)])
#
#         # self.Self_Attn1 = Self_Attn(in_dim=32,  activation=None)
#         # self.Self_Attn2 = Self_Attn(in_dim=64,  activation=None)
#         # self.Self_Attn3 = Self_Attn(in_dim=128, activation=None)
#         self.att = SimAM()
#         # self.att2 = NonLocalBlock(64)
#         # self.att3 = NonLocalBlock(128)
#     def forward(self, x):
#         # 曲线调整
#         x_curve = self.apply_curve(x)
#
#         # 编码器（保存各层特征用于跳跃连接）
#         enc1 = self.enc_conv1(x_curve)  # [B, 32, H, W]
#         enc1_att = self.att(enc1)
#         enc2 = self.enc_conv2(enc1_att)  # [B, 64, H/2, W/2]
#         enc2_att = self.att(enc2)
#         enc3 = self.enc_conv3(enc2_att)  # [B, 128, H/4, W/4]
#         enc3_att = self.att(enc3)
#
#         # 解码器（跳跃连接）
#         dec1 = self.dec_up1(enc3_att)  # [B, 64, H/2, W/2]
#         dec1 = torch.cat([dec1, enc2], dim=1)  # 跳跃连接：拼接enc2
#         dec1 = self.dec_conv1(dec1)  # [B, 64, H/2, W/2]
#
#         dec2 = self.dec_up2(dec1)  # [B, 32, H, W]
#         dec2 = torch.cat([dec2, enc1], dim=1)  # 跳跃连接：拼接enc1
#         dec2 = self.dec_conv2(dec2)  # [B, 32, H, W]
#
#         # 最终输出
#         enhanced = torch.sigmoid(self.final_conv(dec2)) + x
#         # print(enhanced,x)
#         return enhanced
#
#     def apply_curve(self, x):
#         """可学习的光照曲线调整"""   #y=sqrt(1-(x-1)^2)
#         B, C, H, W = x.shape
#         #params = (torch.tanh(self.curve_params) + 1) / 2
#         params = self.curve_params.view(1, -1, 1, 1).expand(B, -1, H, W)
#         return torch.clamp(x + params * (x - x ** 2), 0, 1)

class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()

        # 编码器（下采样）
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 解码器（上采样 + 跳跃连接）
        self.dec_up1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 输入通道=64(上采样)+64(跳跃)
            nn.ReLU()
        )
        self.dec_up2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 输入通道=32(上采样)+32(跳跃)
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

        # 可学习光照曲线（同前）
        self.curve_params = nn.Parameter(torch.ones(3) * 0.5)

        # self.att = SimAM()

    def forward(self, x):
        # 曲线调整
        x_curve = self.apply_curve(x)

        # 编码器（保存各层特征用于跳跃连接）
        enc1 = self.enc_conv1(x_curve)  # [B, 32, H, W]
        # enc1_att = self.att(enc1)
        enc2 = self.enc_conv2(enc1)  # [B, 64, H/2, W/2]
        # enc2_att = self.att(enc2)
        enc3 = self.enc_conv3(enc2)  # [B, 128, H/4, W/4]
        # enc3_att = self.att(enc3)

        # 解码器（跳跃连接）
        dec1 = self.dec_up1(enc3)  # [B, 64, H/2, W/2]
        dec1 = torch.cat([dec1, enc2], dim=1)  # 跳跃连接：拼接enc2
        dec1 = self.dec_conv1(dec1)  # [B, 64, H/2, W/2]

        dec2 = self.dec_up2(dec1)  # [B, 32, H, W]
        dec2 = torch.cat([dec2, enc1], dim=1)  # 跳跃连接：拼接enc1
        dec2 = self.dec_conv2(dec2)  # [B, 32, H, W]

        # 最终输出
        out_fea = torch.sigmoid(self.final_conv(dec2))

        enhanced = torch.clamp(out_fea + x, 0.0001, 1)

        # torch.clamp(illu, 0.0001, 1)
        # print(enhanced,x)
        return enhanced

    def apply_curve(self, x):
        """可学习的光照曲线调整"""  # y=sqrt(1-(x-1)^2)
        B, C, H, W = x.shape
        # params = (torch.tanh(self.curve_params) + 1) / 2
        params = self.curve_params.view(1, -1, 1, 1).expand(B, -1, H, W)
        return torch.clamp(x + params * (x - x ** 2), 0, 1)

    #def small
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True)/n + self.e_lambda)) + 0.5
        return x * self.activation(y)

from utils.torch_utils import select_device
if __name__ == "__main__":
    device = select_device(1)
    # input = torch.rand(8, 128, 40, 40).to(device)
    # # Self_Attn = Self_Attn(128,None).to(device)
    # out = Self_Attn(input)
    # print(out[0].shape, out[1].shape)

    input_efc = (torch.rand(8, 128, 40, 40).to(device), torch.rand(8, 64, 40, 40).to(device))
    EFC_Optimized = EFC_Optimized( c1=128, c2=64).to(device)
    out_efc = EFC_Optimized(input_efc)
    print(out_efc.shape)


    input_tensor = torch.randn(8, 64, 32, 32)
    non_local_block = NonLocalBlock(in_channels=64)
    out_nolocal = non_local_block(input_tensor)
    print(out_nolocal.shape)

    input_imgs = torch.randn(8, 3, 640, 640).to(device)
    Enhance = Enhance().to(device)
    imgs_en = Enhance(input_imgs)
    print(imgs_en.shape)